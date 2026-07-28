"""Evaluate a trained RF-DETR checkpoint on an isolated external COCO test.

Examples:
    python -m src.evaluate --config configs/test_object2.yaml preflight
    python -m src.evaluate --config configs/test_object2.yaml run --train-run runs/Object_1/rfdetr_small/train_<run_id>

External results are stored below ``outputs/external_test`` and never replace
the official ``test_results`` produced by ``src.train``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import traceback
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .train import (
    REPO_ROOT,
    analyze_test_predictions,
    category_mapping,
    checkpoint_audit,
    environment_report,
    load_best_model,
    load_yaml_mapping,
    now_iso,
    read_json,
    release_model,
    resolve_from_repo,
    structured_test_metrics,
    verify_dataset_lock,
    write_json,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "test_object2.yaml"


def configure_utf8_console() -> None:
    """Prevent Rich metric tables from failing on legacy Windows code pages."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def load_external_config(path: Path) -> dict[str, Any]:
    config_path = path.resolve()
    config = load_yaml_mapping(config_path)
    for field in ("dataset_config", "model", "evaluation"):
        if field not in config:
            raise ValueError(f"Missing external-test config field: {field}")

    dataset_path = resolve_from_repo(config["dataset_config"])
    dataset = load_yaml_mapping(dataset_path)
    for field in (
        "name",
        "root",
        "format",
        "splits",
        "annotation_file",
        "lock_file",
        "classes",
    ):
        if field not in dataset:
            raise ValueError(f"Missing external dataset config field: {field}")
    if str(dataset["format"]).lower() != "coco":
        raise ValueError("External test currently supports only COCO format.")
    if not isinstance(dataset["splits"], dict) or "test" not in dataset["splits"]:
        raise ValueError("External dataset must declare splits.test.")
    classes = dataset["classes"]
    if not isinstance(classes, list) or not classes or len(classes) != len(set(classes)):
        raise ValueError("External dataset classes must be a non-empty unique list.")

    dataset["_config_path"] = str(dataset_path)
    dataset["root"] = str(resolve_from_repo(dataset["root"]))
    config["dataset"] = dataset
    config["_config_path"] = str(config_path)
    return config


def external_annotation_path(dataset: dict[str, Any]) -> Path:
    return (
        Path(dataset["root"])
        / str(dataset["splits"]["test"])
        / str(dataset["annotation_file"])
    )


def validate_external_test(dataset: dict[str, Any]) -> dict[str, Any]:
    root = Path(dataset["root"]).resolve()
    split_root = (root / str(dataset["splits"]["test"])).resolve()
    annotation_path = external_annotation_path(dataset)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing external COCO annotation: {annotation_path}")
    data = read_json(annotation_path)
    if not all(key in data for key in ("images", "annotations", "categories")):
        raise ValueError(f"Invalid COCO structure: {annotation_path}")

    classes = list(dataset["classes"])
    category_ids, category_to_index = category_mapping(data, classes, "external_test")
    images_by_id: dict[int, dict[str, Any]] = {}
    file_names: set[str] = set()
    for image in data["images"]:
        image_id = int(image["id"])
        if image_id in images_by_id:
            raise ValueError(f"Duplicate external image id: {image_id}")
        filename = str(image["file_name"])
        if filename in file_names:
            raise ValueError(f"Duplicate external filename: {filename}")
        image_path = (split_root / filename).resolve()
        if not image_path.is_relative_to(split_root):
            raise ValueError(f"Unsafe external image path: {filename}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing external image: {image_path}")
        images_by_id[image_id] = image
        file_names.add(filename)

    annotation_ids: set[int] = set()
    annotated_images: Counter[int] = Counter()
    objects_by_class: Counter[str] = Counter()
    for annotation in data["annotations"]:
        annotation_id = int(annotation["id"])
        image_id = int(annotation["image_id"])
        coco_category_id = int(annotation["category_id"])
        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate external annotation id: {annotation_id}")
        if image_id not in images_by_id:
            raise ValueError(f"Annotation {annotation_id} references an unknown image.")
        if coco_category_id not in category_to_index:
            raise ValueError(f"Annotation {annotation_id} references an unknown class.")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Annotation {annotation_id} has an invalid bbox.")
        x, y, width, height = (float(value) for value in bbox)
        if (
            not all(math.isfinite(value) for value in (x, y, width, height))
            or x < 0
            or y < 0
            or width <= 0
            or height <= 0
        ):
            raise ValueError(f"Annotation {annotation_id} has invalid bbox {bbox}.")
        image = images_by_id[image_id]
        if x + width > float(image["width"]) + 1:
            raise ValueError(f"Annotation {annotation_id} exceeds image width.")
        if y + height > float(image["height"]) + 1:
            raise ValueError(f"Annotation {annotation_id} exceeds image height.")
        annotation_ids.add(annotation_id)
        annotated_images[image_id] += 1
        class_name = classes[category_to_index[coco_category_id]]
        objects_by_class[class_name] += 1

    manifest_path = root / str(dataset.get("manifest_file", "test_manifest.json"))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing external test manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    expected_counts = manifest.get("counts", {})
    actual_counts = {
        "images": len(images_by_id),
        "annotations": len(annotation_ids),
        "negative_images": sum(
            annotated_images[image_id] == 0 for image_id in images_by_id
        ),
        "objects_by_class": {
            class_name: objects_by_class[class_name] for class_name in classes
        },
    }
    if expected_counts != actual_counts:
        raise ValueError(
            "External COCO counts differ from test_manifest.json: "
            f"expected {expected_counts}, got {actual_counts}"
        )
    separation = manifest.get("separation", {})
    if separation != {
        "included_in_training": False,
        "included_in_validation": False,
        "replaces_official_test": False,
    }:
        raise ValueError("External test manifest does not guarantee split separation.")

    return {
        "status": "PASS",
        "root": str(root),
        "annotation_file": str(annotation_path),
        "manifest_file": str(manifest_path),
        "category_ids": category_ids,
        **actual_counts,
        "separation": separation,
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    device = str(config["evaluation"].get("device", "cuda")).lower()
    return {
        "status": "PASS",
        "checked_at": now_iso(),
        "config": config["_config_path"],
        "dataset_config": config["dataset"]["_config_path"],
        "dataset_lock": verify_dataset_lock(config["dataset"]),
        "dataset": validate_external_test(config["dataset"]),
        "environment": environment_report(require_cuda=device.startswith("cuda")),
        "mapping": {
            index: name for index, name in enumerate(config["dataset"]["classes"])
        },
    }


def resolve_checkpoint(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.train_run is not None:
        train_run = args.train_run.resolve()
        checkpoint = train_run / "checkpoint_best_total.pth"
        if not (train_run / "final_report.json").is_file():
            raise FileNotFoundError(
                f"Training run is not complete (missing final_report.json): {train_run}"
            )
        return checkpoint, train_run
    return args.checkpoint.resolve(), None


def create_output_dir(
    config: dict[str, Any], train_run: Path | None
) -> Path:
    dataset_name = str(config["dataset"]["name"])
    safe_dataset = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in dataset_name
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    leaf = f"eval_{stamp}_{uuid.uuid4().hex[:6]}"
    root = REPO_ROOT / "outputs" / "external_test" / safe_dataset
    output = root / leaf
    output.mkdir(parents=True, exist_ok=False)
    return output


def prepare_rfdetr_external_view(
    dataset: dict[str, Any], output_dir: Path
) -> Path:
    """Create the minimal Roboflow layout RF-DETR requires for test-only data.

    RF-DETR detects COCO format solely from train/_annotations.coco.json even
    when evaluate(split="test") only builds the test dataset. The empty train
    annotation below is a format marker, not training data.
    """
    source_test = Path(dataset["root"]) / str(dataset["splits"]["test"])
    source_coco = read_json(source_test / str(dataset["annotation_file"]))
    view_root = output_dir / "_dataset_view"
    train_dir = view_root / "train"
    test_dir = view_root / "test"
    train_dir.mkdir(parents=True, exist_ok=False)
    test_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        train_dir / "_annotations.coco.json",
        {
            "info": {
                "description": "Empty RF-DETR format marker; not used for training."
            },
            "licenses": source_coco.get("licenses", []),
            "images": [],
            "annotations": [],
            "categories": source_coco["categories"],
        },
    )
    files = [
        (
            source_test / str(dataset["annotation_file"]),
            test_dir / "_annotations.coco.json",
        )
    ]
    files.extend(
        (
            source_test / str(image["file_name"]),
            test_dir / str(image["file_name"]),
        )
        for image in source_coco["images"]
    )
    for source, target in files:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return view_root


def run_external_test(
    config: dict[str, Any], checkpoint: Path, train_run: Path | None
) -> Path:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    report = preflight(config)
    dataset = config["dataset"]
    classes = list(dataset["classes"])
    audit = checkpoint_audit(checkpoint, classes)
    output_dir = create_output_dir(config, train_run)
    write_json(output_dir / "preflight_report.json", report)
    write_json(output_dir / "resolved_config.json", config)
    write_json(output_dir / "checkpoint_audit.json", audit)
    write_json(
        output_dir / "evaluation_manifest.json",
        {
            "status": "RUNNING",
            "started_at": now_iso(),
            "external_dataset": dataset["name"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": audit["sha256"],
            "source_train_run": str(train_run) if train_run else None,
            "official_test_results_modified": False,
        },
    )

    rfdetr_dataset_dir = prepare_rfdetr_external_view(dataset, output_dir)
    model = None
    try:
        model = load_best_model(checkpoint, config["model"]["name"], classes)
        evaluation = config["evaluation"]
        raw_metrics = model.evaluate(
            split="test",
            dataset_dir=str(rfdetr_dataset_dir),
            output_dir=str(output_dir / "_eval_runtime"),
            dataset_file="roboflow",
            class_names=classes,
            device=evaluation.get("device", "cuda"),
            batch_size=int(evaluation.get("batch_size", 4)),
            num_workers=int(evaluation.get("num_workers", 0)),
            log_per_class_metrics=True,
            progress_bar=evaluation.get("progress_bar", "tqdm"),
        )
        metrics = structured_test_metrics(raw_metrics, classes)
        metrics.update(
            {
                "evaluated_at": now_iso(),
                "external_dataset": dataset["name"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": audit["sha256"],
                "images": report["dataset"]["images"],
                "annotations": report["dataset"]["annotations"],
                "negative_images": report["dataset"]["negative_images"],
            }
        )
        write_json(output_dir / "metrics.json", metrics)

        analysis = analyze_test_predictions(
            model,
            dataset,
            output_dir,
            float(evaluation.get("confidence_threshold", 0.25)),
            float(evaluation.get("match_iou_threshold", 0.50)),
        )
        write_json(output_dir / "analysis_summary.json", analysis)
        final_report = {
            "status": "PASS",
            "finished_at": now_iso(),
            "external_dataset": dataset["name"],
            "checkpoint": str(checkpoint),
            "source_train_run": str(train_run) if train_run else None,
            "official_test_results_modified": False,
            "metrics": metrics,
            "error_analysis": analysis,
            "artifacts": {
                "confusion_matrix": str(output_dir / "confusion_matrix.png"),
                "false_positive_false_negative": str(
                    output_dir / "false_positive_false_negative"
                ),
                "predictions": str(output_dir / "predictions"),
                "coco_predictions": str(output_dir / "test_predictions.coco.json"),
            },
        }
        write_json(output_dir / "final_report.json", final_report)
        write_json(
            output_dir / "evaluation_manifest.json",
            {
                "status": "COMPLETE",
                "started_at": read_json(
                    output_dir / "evaluation_manifest.json"
                )["started_at"],
                "finished_at": now_iso(),
                "external_dataset": dataset["name"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": audit["sha256"],
                "source_train_run": str(train_run) if train_run else None,
                "official_test_results_modified": False,
            },
        )
        print(f"EXTERNAL_TEST_RUN={output_dir}")
        print(f"FINAL_REPORT={output_dir / 'final_report.json'}")
        return output_dir
    except Exception as exc:
        manifest = read_json(output_dir / "evaluation_manifest.json")
        manifest.update(
            {
                "status": "FAILED",
                "failed_at": now_iso(),
                "error": str(exc),
            }
        )
        write_json(output_dir / "evaluation_manifest.json", manifest)
        write_json(
            output_dir / "failure.json",
            {
                "failed_at": now_iso(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if model is not None:
            release_model(model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"External-test YAML (default: {DEFAULT_CONFIG})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="Verify the external dataset and GPU.")
    run = subparsers.add_parser(
        "run", help="Evaluate a completed checkpoint without touching official test results."
    )
    checkpoint_source = run.add_mutually_exclusive_group(required=True)
    checkpoint_source.add_argument(
        "--train-run",
        type=Path,
        help="Completed training run containing checkpoint_best_total.pth.",
    )
    checkpoint_source.add_argument(
        "--checkpoint",
        type=Path,
        help="Explicit checkpoint_best_total.pth path.",
    )
    return parser


def main() -> int:
    configure_utf8_console()
    args = build_parser().parse_args()
    config = load_external_config(args.config)
    if args.command == "preflight":
        report = preflight(config)
        safe_dataset = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(config["dataset"]["name"])
        )
        output = (
            REPO_ROOT
            / "outputs"
            / "dataset_audit"
            / safe_dataset
            / "preflight.json"
        )
        write_json(output, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"PREFLIGHT_REPORT={output}")
    elif args.command == "run":
        checkpoint, train_run = resolve_checkpoint(args)
        run_external_test(config, checkpoint, train_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
