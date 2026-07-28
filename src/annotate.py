"""Single CLI entrypoint for the standalone pretrained-SAM3 annotation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .sam3_pipeline.coco_exporter import export_coco
from .sam3_pipeline.predictor import Sam3Predictor, propose_dataset
from .sam3_pipeline.reporter import write_review_report
from .sam3_pipeline.reviewer import (
    create_previews,
    review_dataset,
    validate_preview_output,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sam3.yaml"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _resolve_under(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    payload = _load_mapping(path.resolve())
    required = {"dataset_config", "model", "input", "postprocess", "output", "runtime"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Missing sam3 config sections: {missing}")

    dataset_path = _resolve(payload["dataset_config"])
    dataset = _load_mapping(dataset_path)
    classes = dataset.get("classes")
    if (
        not isinstance(classes, list)
        or not classes
        or len(classes) != len(set(classes))
    ):
        raise ValueError("dataset_config classes must be a non-empty unique list.")
    dataset["_config_path"] = str(dataset_path)
    payload["dataset"] = dataset
    payload["dataset_config"] = str(dataset_path)

    for section in ("model", "input", "postprocess", "output", "runtime"):
        if not isinstance(payload[section], dict):
            raise ValueError(f"{section} must be a mapping.")

    model = payload["model"]
    for field in ("source", "checkpoint", "device"):
        if field not in model:
            raise ValueError(f"Missing model.{field}")
    source = str(model["source"]).strip().lower()
    if source not in {"huggingface", "local"}:
        raise ValueError("model.source must be 'huggingface' or 'local'.")
    model["source"] = source
    if source == "huggingface":
        repo_id = str(model.get("repo_id", "")).strip()
        if not repo_id:
            raise ValueError("model.repo_id is required for Hugging Face.")
        model["repo_id"] = repo_id
    elif model["checkpoint"] is None:
        raise ValueError("model.checkpoint is required when model.source is local.")
    if model["checkpoint"] is not None:
        model["checkpoint"] = str(_resolve(model["checkpoint"]))

    input_config = payload["input"]
    for field in ("images_dir", "extensions"):
        if field not in input_config:
            raise ValueError(f"Missing input.{field}")
    extensions = input_config["extensions"]
    if not isinstance(extensions, list) or not extensions:
        raise ValueError("input.extensions must be a non-empty list.")
    normalized_extensions = []
    for extension in extensions:
        value = str(extension).strip().lower()
        if not value.startswith("."):
            value = f".{value}"
        if value not in normalized_extensions:
            normalized_extensions.append(value)
    input_config["extensions"] = normalized_extensions
    input_config["images_dir"] = str(_resolve(input_config["images_dir"]))

    postprocess = payload["postprocess"]
    required_postprocess = {
        "mask_threshold",
        "min_mask_area",
        "duplicate_iou_threshold",
        "valid_region",
        "valid_region_min_fraction",
    }
    missing_postprocess = sorted(required_postprocess - postprocess.keys())
    if missing_postprocess:
        raise ValueError(f"Missing postprocess fields: {missing_postprocess}")
    for field in (
        "mask_threshold",
        "duplicate_iou_threshold",
        "valid_region_min_fraction",
    ):
        if not 0 <= float(postprocess[field]) <= 1:
            raise ValueError(f"postprocess.{field} must be between 0 and 1.")
    if int(postprocess["min_mask_area"]) < 1:
        raise ValueError("postprocess.min_mask_area must be positive.")

    output = payload["output"]
    required_output = {
        "root",
        "proposals_dir",
        "previews_dir",
        "review_file",
        "labeling_report",
        "summary_file",
        "annotation_file",
    }
    missing_output = sorted(required_output - output.keys())
    if missing_output:
        raise ValueError(f"Missing output fields: {missing_output}")
    output_root = _resolve(output["root"])
    output["root"] = str(output_root)
    for field in (
        "proposals_dir",
        "previews_dir",
        "review_file",
        "labeling_report",
        "summary_file",
    ):
        resolved = _resolve_under(output_root, output[field])
        if resolved != output_root and output_root not in resolved.parents:
            raise ValueError(f"output.{field} must stay inside output.root.")
        output[field] = str(resolved)
    annotation_path = _resolve(output["annotation_file"])
    input_dataset_root = Path(input_config["images_dir"]).parent
    if (
        annotation_path != input_dataset_root
        and input_dataset_root not in annotation_path.parents
    ):
        raise ValueError(
            "output.annotation_file must stay inside the input dataset root."
        )
    output["annotation_file"] = str(annotation_path)

    overwrite = payload["runtime"].get("overwrite")
    if not isinstance(overwrite, bool):
        raise ValueError("runtime.overwrite must be true or false.")
    payload["_repo_root"] = str(REPO_ROOT)
    payload["_config_path"] = str(path.resolve())
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("propose", "review", "export"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--config", type=Path, default=DEFAULT_CONFIG, help="SAM3 YAML config"
        )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "propose":
        validate_preview_output(config)
        predictor = Sam3Predictor(config)
        manifest = propose_dataset(config, predictor)
        create_previews(config, manifest)
        print(
            f"PROPOSALS={manifest['proposal_count']} "
            f"MANIFEST={Path(config['output']['proposals_dir']) / 'manifest.json'}"
        )
    elif args.command == "review":
        review = review_dataset(config)
        report = write_review_report(config, review)
        print(
            f"REVIEW={review['status']} "
            f"SUMMARY={config['output']['summary_file']} "
            f"STATUS={report['status']}"
        )
    elif args.command == "export":
        annotation_path = export_coco(config)
        print(f"COCO={annotation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
