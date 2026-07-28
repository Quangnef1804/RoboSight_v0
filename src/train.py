"""Reusable RF-DETR training and evaluation pipeline for COCO datasets.

Commands:
    preflight       Verify the locked dataset and local training environment.
    smoke           Fine-tune for two epochs and create visual/automatic checks.
    finalize-smoke  Recover smoke post-processing without training again.
    approve-smoke   Record that a human inspected the smoke predictions.
    train           Start a fresh main run and evaluate its best checkpoint once.

The script intentionally keeps the test split out of training and smoke testing.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, ImageColor, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "train.yaml"
SPLIT_ROLES = ("train", "valid", "test")
BACKGROUND = "__background__"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_from_repo(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def load_config(path: Path) -> dict[str, Any]:
    """Load train.yaml, then resolve its independent dataset.yaml."""
    train_path = path.resolve()
    config = load_yaml_mapping(train_path)
    for section in ("dataset_config", "model", "training"):
        if section not in config:
            raise ValueError(f"Missing train config section: {section}")

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
            raise ValueError(f"Missing dataset config field: {field}")
    if str(dataset["format"]).lower() != "coco":
        raise ValueError(f"Only COCO datasets are supported, got {dataset['format']!r}.")
    if not isinstance(dataset["splits"], dict):
        raise ValueError("dataset.splits must be a mapping.")
    missing_splits = [role for role in SPLIT_ROLES if role not in dataset["splits"]]
    if missing_splits:
        raise ValueError(f"dataset.splits is missing: {missing_splits}")
    classes = dataset["classes"]
    if not isinstance(classes, list) or not classes or len(classes) != len(set(classes)):
        raise ValueError("dataset.classes must be a non-empty list of unique names.")

    dataset["_config_path"] = str(dataset_path)
    dataset["root"] = str(resolve_from_repo(dataset["root"]))
    config["dataset"] = dataset
    config["runs_root"] = str(resolve_from_repo(config.get("runs_root", "runs")))
    config["_config_path"] = str(train_path)
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset_lock(dataset: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(dataset["root"])
    lock_path = dataset_dir / str(dataset["lock_file"])
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing dataset lock: {lock_path}")
    lock = read_json(lock_path)
    if lock.get("locked") is not True or lock.get("validator_status") != "PASS":
        raise ValueError("Dataset lock is not marked locked/PASS.")

    expected_hashes = lock.get("sha256")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise ValueError("Dataset lock contains no sha256 inventory.")
    if lock.get("file_count") != len(expected_hashes):
        raise ValueError(
            f"Lock file_count={lock.get('file_count')} but contains {len(expected_hashes)} hashes."
        )

    missing: list[str] = []
    changed: list[dict[str, str]] = []
    for relative, expected in expected_hashes.items():
        target = dataset_dir / relative
        if not target.is_file():
            missing.append(relative)
            continue
        actual = sha256_file(target)
        if actual.lower() != str(expected).lower():
            changed.append({"file": relative, "expected": expected, "actual": actual})

    if missing or changed:
        raise ValueError(
            "Locked dataset integrity check failed: "
            f"{len(missing)} missing, {len(changed)} changed. "
            f"Examples: {(missing + [row['file'] for row in changed])[:5]}"
        )
    return {
        "status": "PASS",
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "locked_at": lock.get("locked_at"),
        "validator_status": lock.get("validator_status"),
        "near_duplicate_audit": lock.get("near_duplicate_audit"),
        "verified_file_count": len(expected_hashes),
    }


def split_directory(dataset: dict[str, Any], role: str) -> Path:
    return Path(dataset["root"]) / str(dataset["splits"][role])


def load_coco_split(dataset: dict[str, Any], role: str) -> dict[str, Any]:
    path = split_directory(dataset, role) / str(dataset["annotation_file"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing COCO annotation: {path}")
    data = read_json(path)
    if not all(key in data for key in ("images", "annotations", "categories")):
        raise ValueError(f"Invalid COCO structure: {path}")
    return data


def category_mapping(
    data: dict[str, Any], expected_classes: list[str], role: str
) -> tuple[list[int], dict[int, int]]:
    categories = sorted(data["categories"], key=lambda item: int(item["id"]))
    category_ids = [int(item["id"]) for item in categories]
    category_names = [str(item["name"]) for item in categories]
    if len(category_ids) != len(set(category_ids)):
        raise ValueError(f"{role}: duplicate COCO category id.")
    if category_names != expected_classes:
        raise ValueError(
            f"{role}: class order mismatch; expected {expected_classes}, got {category_names}"
        )
    return category_ids, {
        category_id: class_index for class_index, category_id in enumerate(category_ids)
    }


def validate_coco_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(dataset["root"])
    expected_classes = list(dataset["classes"])
    report: dict[str, Any] = {"status": "PASS", "splits": {}}
    all_filenames: dict[str, str] = {}
    reference_category_ids: list[int] | None = None

    for role in SPLIT_ROLES:
        data = load_coco_split(dataset, role)
        category_ids, category_to_index = category_mapping(data, expected_classes, role)
        if reference_category_ids is None:
            reference_category_ids = category_ids
        elif category_ids != reference_category_ids:
            raise ValueError(
                f"{role}: category ids differ across splits; "
                f"expected {reference_category_ids}, got {category_ids}"
            )

        images = data["images"]
        annotations = data["annotations"]
        image_by_id: dict[int, dict[str, Any]] = {}
        for image in images:
            image_id = int(image["id"])
            if image_id in image_by_id:
                raise ValueError(f"{role}: duplicate image id {image_id}")
            image_by_id[image_id] = image
            filename = str(image["file_name"])
            split_root = split_directory(dataset, role).resolve()
            image_path = (split_root / filename).resolve()
            if not image_path.is_relative_to(split_root):
                raise ValueError(f"{role}: unsafe image path outside split: {filename}")
            if not image_path.is_file():
                raise FileNotFoundError(f"{role}: missing image {image_path}")
            if filename in all_filenames:
                raise ValueError(
                    f"Image appears in multiple splits: {filename} "
                    f"({all_filenames[filename]}, {role})"
                )
            all_filenames[filename] = role

        annotation_ids: set[int] = set()
        annotations_by_image: Counter[int] = Counter()
        objects_by_class: Counter[str] = Counter()
        for annotation in annotations:
            annotation_id = int(annotation["id"])
            if annotation_id in annotation_ids:
                raise ValueError(f"{role}: duplicate annotation id {annotation_id}")
            annotation_ids.add(annotation_id)

            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            if image_id not in image_by_id:
                raise ValueError(f"{role}: annotation references unknown image {image_id}")
            if category_id not in category_to_index:
                raise ValueError(f"{role}: annotation references unknown category {category_id}")

            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"{role}: annotation {annotation_id} has invalid bbox")
            x, y, width, height = (float(value) for value in bbox)
            if not all(math.isfinite(value) for value in (x, y, width, height)):
                raise ValueError(f"{role}: annotation {annotation_id} has non-finite bbox")
            if width <= 0 or height <= 0 or x < 0 or y < 0:
                raise ValueError(f"{role}: annotation {annotation_id} has invalid bbox {bbox}")
            image = image_by_id[image_id]
            image_width = float(image["width"])
            image_height = float(image["height"])
            if x + width > image_width + 1 or y + height > image_height + 1:
                raise ValueError(
                    f"{role}: annotation {annotation_id} bbox exceeds image bounds"
                )

            annotations_by_image[image_id] += 1
            objects_by_class[expected_classes[category_to_index[category_id]]] += 1

        report["splits"][role] = {
            "directory": str(dataset["splits"][role]),
            "images": len(images),
            "annotations": len(annotations),
            "negative_images": sum(
                annotations_by_image[image_id] == 0 for image_id in image_by_id
            ),
            "objects_by_class": {
                class_name: objects_by_class[class_name] for class_name in expected_classes
            },
            "categories": [
                {"id": category_id, "name": class_name}
                for category_id, class_name in zip(category_ids, expected_classes)
            ],
        }

    report["total_images"] = len(all_filenames)
    report["total_negative_images"] = sum(
        row["negative_images"] for row in report["splits"].values()
    )
    report["category_ids"] = reference_category_ids

    manifest_path = dataset_dir / str(dataset.get("manifest_file", "split_manifest.json"))
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        for role in SPLIT_ROLES:
            expected = manifest["splits"][role]
            actual = report["splits"][role]
            if actual["images"] != expected["image_count"]:
                raise ValueError(f"{role}: image count differs from split manifest")
            if actual["annotations"] != expected["annotation_count"]:
                raise ValueError(f"{role}: annotation count differs from split manifest")
            if actual["objects_by_class"] != expected["objects_by_class"]:
                raise ValueError(f"{role}: class counts differ from split manifest")
            if (
                "negative_sample_count" in expected
                and actual["negative_images"] != expected["negative_sample_count"]
            ):
                raise ValueError(f"{role}: negative count differs from split manifest")
        report["split_manifest_status"] = "PASS"
        report["split_manifest_path"] = str(manifest_path)
        report["near_duplicate_status"] = manifest.get("near_duplicate_policy", {}).get("status")
        report["class_balance_status"] = manifest.get("class_balance_policy", {}).get("status")
    else:
        report["split_manifest_status"] = "NOT_PRESENT"
    return report


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def rf_detr_git_commit() -> str | None:
    source_dir = REPO_ROOT / "third-party" / "RF-DETR"
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_report(require_cuda: bool = True) -> dict[str, Any]:
    import torch

    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required by config, but torch.cuda.is_available() is False.")
    report = {
        "status": "PASS",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_memory_gib": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if cuda_available
            else None
        ),
        "rfdetr": package_version("rfdetr"),
        "rfdetr_source_commit": rf_detr_git_commit(),
        "pytorch_lightning": package_version("pytorch-lightning"),
        "torchmetrics": package_version("torchmetrics"),
        "supervision": package_version("supervision"),
        "pycocotools": package_version("pycocotools"),
    }
    if report["rfdetr"] is None or report["pytorch_lightning"] is None:
        raise RuntimeError(
            "RF-DETR training dependencies are missing. Install requirements-rfdetr.txt."
        )
    return report


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    classes = list(dataset["classes"])
    device = str(config["training"].get("device", "cuda")).lower()
    report = {
        "status": "PASS",
        "checked_at": now_iso(),
        "train_config": config["_config_path"],
        "dataset_config": dataset["_config_path"],
        "dataset_lock": verify_dataset_lock(dataset),
        "dataset": validate_coco_dataset(dataset),
        "environment": environment_report(require_cuda=device.startswith("cuda")),
        "mapping": {index: name for index, name in enumerate(classes)},
    }
    return report


def create_run_dir(config: dict[str, Any], phase: str) -> Path:
    root = (
        Path(config["runs_root"])
        / str(config["dataset"]["name"])
        / str(config["model"]["name"])
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{phase}_{stamp}_{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def prepare_rfdetr_dataset_view(dataset: dict[str, Any], run_dir: Path) -> Path:
    """Return a Roboflow-layout view understood by RF-DETR.

    The original root is reused when it already uses train/valid/test and
    _annotations.coco.json. Otherwise a run-local view is made with hardlinks
    (copy fallback), so the source dataset remains untouched.
    """
    standard_splits = {"train": "train", "valid": "valid", "test": "test"}
    if (
        dataset["splits"] == standard_splits
        and str(dataset["annotation_file"]) == "_annotations.coco.json"
    ):
        return Path(dataset["root"])

    view_root = run_dir / "_dataset_view"
    for role in SPLIT_ROLES:
        target_split = view_root / standard_splits[role]
        target_split.mkdir(parents=True, exist_ok=False)
        source_annotation = (
            split_directory(dataset, role) / str(dataset["annotation_file"])
        )
        data = load_coco_split(dataset, role)
        files = [(source_annotation, target_split / "_annotations.coco.json")]
        files.extend(
            (
                split_directory(dataset, role) / str(image["file_name"]),
                target_split / str(image["file_name"]),
            )
            for image in data["images"]
        )
        for source, target in files:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
    return view_root


def update_run_manifest(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    payload = read_json(path) if path.exists() else {}
    payload.update(updates)
    write_json(path, payload)
    return payload


def annotations_by_image(data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        grouped[int(annotation["image_id"])].append(annotation)
    return grouped


def xywh_to_xyxy(box: Iterable[float]) -> list[float]:
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
) -> None:
    font = ImageFont.load_default()
    left, top = int(xy[0]), int(xy[1])
    bbox = draw.textbbox((left, top), text, font=font)
    draw.rectangle(bbox, fill=color)
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    draw.text((left, top), text, fill=(255, 255, 255) if luminance < 130 else (0, 0, 0), font=font)


def class_color(class_name: str) -> tuple[int, int, int]:
    """Use a CSS color name when possible, otherwise derive a stable color."""
    try:
        return ImageColor.getrgb(class_name)
    except ValueError:
        hue = int(hashlib.sha256(class_name.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.88)
        return int(red * 255), int(green * 255), int(blue * 255)


def draw_boxes(
    image_path: Path,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    classes: list[str],
    output_path: Path,
    error_gt: set[int] | None = None,
    error_pred: set[int] | None = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    error_gt = error_gt or set()
    error_pred = error_pred or set()

    for index, item in enumerate(ground_truth):
        class_id = int(item["class_id"])
        class_name = classes[class_id]
        box = item["xyxy"]
        color = (255, 80, 0) if index in error_gt else class_color(class_name)
        for offset in range(3):
            draw.rectangle(
                [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
                outline=color,
            )
        prefix = "FN/MIS " if index in error_gt else "GT "
        draw_label(draw, (box[0], max(0, box[1] - 13)), prefix + class_name, color)

    for index, item in enumerate(predictions):
        class_id = int(item["class_id"])
        class_name = classes[class_id]
        box = item["xyxy"]
        color = (0, 190, 255) if index in error_pred else class_color(class_name)
        draw.rectangle(box, outline=color, width=2)
        prefix = "FP/MIS " if index in error_pred else "P "
        draw_label(
            draw,
            (box[0], min(image.height - 13, box[1] + 2)),
            f"{prefix}{class_name} {float(item['score']):.2f}",
            color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=94)


def save_ground_truth_previews(
    dataset: dict[str, Any],
    role: str,
    output_dir: Path,
    limit: int | None,
) -> int:
    classes = list(dataset["classes"])
    data = load_coco_split(dataset, role)
    _, category_to_index = category_mapping(data, classes, role)
    grouped = annotations_by_image(data)
    count = 0
    image_rows = sorted(data["images"], key=lambda row: row["file_name"])
    if limit is not None:
        image_rows = image_rows[:limit]
    for image_info in image_rows:
        image_id = int(image_info["id"])
        ground_truth = [
            {
                "xyxy": xywh_to_xyxy(annotation["bbox"]),
                "class_id": category_to_index[int(annotation["category_id"])],
            }
            for annotation in grouped.get(image_id, [])
        ]
        filename = str(image_info["file_name"])
        draw_boxes(
            split_directory(dataset, role) / filename,
            ground_truth,
            [],
            classes,
            output_dir / filename,
        )
        count += 1
    return count


def model_class(variant: str) -> type:
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    variants = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    normalized = variant.lower().removeprefix("rfdetr_").removeprefix("rf-detr-")
    try:
        return variants[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported model variant: {variant}") from exc


def release_model(model: Any) -> None:
    """Release an in-memory model before loading the selected checkpoint."""
    try:
        model.remove_optimized_model()
    except (AttributeError, RuntimeError):
        pass
    try:
        model.model.model = model.model.model.to("cpu")
        model.model.device = "cpu"
    except (AttributeError, RuntimeError):
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def train_kwargs(
    config: dict[str, Any],
    phase: str,
    run_dir: Path,
    rfdetr_dataset_dir: Path | None = None,
) -> dict[str, Any]:
    common = dict(config["training"])
    if phase == "smoke":
        phase_config = {
            "epochs": 2,
            "warmup_epochs": 0.0,
            "checkpoint_interval": 1,
            "early_stopping": False,
            "save_dataset_grids": True,
            **dict(config.get("smoke", {})),
        }
    elif phase == "train":
        phase_config = {
            **common,
            "save_dataset_grids": False,
        }
    else:
        raise ValueError(f"Unknown training phase: {phase}")
    dataset = config["dataset"]
    kwargs = {
        "dataset_dir": str(rfdetr_dataset_dir or dataset["root"]),
        "output_dir": str(run_dir),
        "dataset_file": "roboflow",
        "class_names": list(dataset["classes"]),
        "run": run_dir.name,
        "project": f"{dataset['name']}_{config['model']['name']}",
        "run_test": False,
        "dont_save_weights": False,
        "compute_train_metrics": False,
        "compute_val_loss": True,
        "train_log_on_step": False,
        "seed": int(common.get("seed", 42)),
        "device": common.get("device", "cuda"),
        "batch_size": int(common.get("batch_size", 4)),
        "grad_accum_steps": int(common.get("grad_accum_steps", 4)),
        "num_workers": int(common.get("num_workers", 0)),
        "lr": float(common.get("lr", 1e-4)),
        "lr_encoder": float(common.get("lr_encoder", 1.5e-4)),
        "weight_decay": float(common.get("weight_decay", 1e-4)),
        "warmup_epochs": float(
            phase_config.get("warmup_epochs", common.get("warmup_epochs", 0.0))
        ),
        "lr_scheduler": common.get("lr_scheduler", "step"),
        "lr_scheduler_kwargs": dict(common.get("lr_scheduler_kwargs", {})),
        "amp_dtype": common.get("amp_dtype", "auto"),
        "use_ema": bool(common.get("use_ema", True)),
        "ema_decay": float(common.get("ema_decay", 0.993)),
        "ema_tau": int(common.get("ema_tau", 100)),
        "ema_update_interval": int(common.get("ema_update_interval", 1)),
        "eval_interval": int(common.get("eval_interval", 1)),
        "log_per_class_metrics": bool(common.get("log_per_class_metrics", True)),
        "tensorboard": bool(common.get("tensorboard", True)),
        "progress_bar": common.get("progress_bar", "tqdm"),
        "multi_scale": bool(common.get("multi_scale", False)),
        "expanded_scales": bool(common.get("expanded_scales", False)),
        "scale_jitter": bool(common.get("scale_jitter", True)),
        "augmentation_backend": common.get("augmentation_backend", "torchvision"),
        "aug_config": common.get("aug_config"),
        "epochs": int(phase_config.get("epochs", common["epochs"])),
        "checkpoint_interval": int(
            phase_config.get("checkpoint_interval", common.get("checkpoint_interval", 5))
        ),
        "early_stopping": bool(
            phase_config.get("early_stopping", common.get("early_stopping", False))
        ),
        "save_dataset_grids": bool(phase_config.get("save_dataset_grids", False)),
    }
    if kwargs["early_stopping"]:
        kwargs.update(
            early_stopping_patience=int(
                phase_config.get(
                    "early_stopping_patience", common.get("early_stopping_patience", 10)
                )
            ),
            early_stopping_min_delta=float(
                phase_config.get(
                    "early_stopping_min_delta", common.get("early_stopping_min_delta", 0.001)
                )
            ),
            early_stopping_use_ema=bool(
                phase_config.get(
                    "early_stopping_use_ema",
                    common.get("early_stopping_use_ema", common.get("use_ema", True)),
                )
            ),
        )
    return kwargs


def checkpoint_audit(checkpoint_path: Path, expected_classes: list[str]) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    head_shapes = {
        key: list(value.shape)
        for key, value in state_dict.items()
        if hasattr(value, "shape") and "class_embed" in key and key.endswith(("weight", "bias"))
    }
    args = checkpoint.get("args")
    if isinstance(args, dict):
        saved_classes = args.get("class_names")
        num_classes = args.get("num_classes")
    else:
        saved_classes = getattr(args, "class_names", None)
        num_classes = getattr(args, "num_classes", None)
    if saved_classes is not None:
        saved_classes = list(saved_classes)
    head_class_counts = sorted(
        {
            int(shape[0]) - 1
            for key, shape in head_shapes.items()
            if key.endswith("weight") and len(shape) >= 1
        }
    )
    if num_classes is None:
        if len(head_class_counts) != 1:
            raise ValueError(
                "Checkpoint metadata has no num_classes and detection heads "
                f"do not imply one unambiguous class count: {head_class_counts}"
            )
        effective_num_classes = head_class_counts[0]
        num_classes_source = "class_head_shape"
    else:
        effective_num_classes = int(num_classes)
        num_classes_source = "args.num_classes"
    report = {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "best_total_source": checkpoint.get("best_total_source"),
        "saved_classes": saved_classes,
        "num_classes": effective_num_classes,
        "num_classes_metadata": num_classes,
        "num_classes_source": num_classes_source,
        "class_head_tensors": head_shapes,
    }
    if saved_classes != expected_classes:
        raise ValueError(
            f"Checkpoint class mapping mismatch: expected {expected_classes}, got {saved_classes}"
        )
    if effective_num_classes != len(expected_classes):
        raise ValueError(
            "Checkpoint num_classes mismatch: "
            f"expected {len(expected_classes)}, got {effective_num_classes} "
            f"(source: {num_classes_source})"
        )
    if not head_shapes:
        raise ValueError("Could not find class_embed tensors in checkpoint.")
    invalid_heads = [
        (key, shape)
        for key, shape in head_shapes.items()
        if key.endswith("weight") and shape[0] != len(expected_classes) + 1
    ]
    if invalid_heads:
        raise ValueError(f"Unexpected detection head shape(s): {invalid_heads}")
    return report


def load_best_model(checkpoint_path: Path, variant: str, expected_classes: list[str]) -> Any:
    cls = model_class(variant)
    model = cls.from_checkpoint(checkpoint_path, trust_checkpoint=True)
    if list(model.class_names) != expected_classes:
        raise ValueError(
            f"Loaded model mapping mismatch: expected {expected_classes}, got {model.class_names}"
        )
    if int(model.model_config.num_classes) != len(expected_classes):
        raise ValueError(
            f"Loaded model has num_classes={model.model_config.num_classes}, expected {len(expected_classes)}"
        )
    return model


def metrics_csv_path(run_dir: Path) -> Path:
    direct = run_dir / "metrics.csv"
    if direct.is_file():
        return direct
    candidates = sorted(run_dir.rglob("metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv found in {run_dir}")
    return candidates[-1]


def metric_history(run_dir: Path) -> dict[str, Any]:
    import pandas as pd

    path = metrics_csv_path(run_dir)
    frame = pd.read_csv(path)
    history: dict[str, Any] = {
        "metrics_csv": str(path),
        "columns": list(frame.columns),
        "train_loss_by_epoch": {},
        "val_map_50_95_by_epoch": {},
        "val_ema_map_50_95_by_epoch": {},
        "per_class_val_ap": {},
    }

    def epoch_values(column: str) -> dict[str, float]:
        if column not in frame.columns or "epoch" not in frame.columns:
            return {}
        subset = frame[["epoch", column]].dropna()
        if subset.empty:
            return {}
        grouped = subset.groupby("epoch")[column].last()
        return {str(int(epoch)): float(value) for epoch, value in grouped.items()}

    loss_column = "train/loss_epoch" if "train/loss_epoch" in frame.columns else "train/loss"
    history["loss_column"] = loss_column
    history["train_loss_by_epoch"] = epoch_values(loss_column)
    history["val_map_50_95_by_epoch"] = epoch_values("val/mAP_50_95")
    history["val_ema_map_50_95_by_epoch"] = epoch_values("val/ema_mAP_50_95")
    for column in frame.columns:
        if column.startswith("val/AP/"):
            history["per_class_val_ap"][column.removeprefix("val/AP/")] = epoch_values(column)
    loss_values = list(history["train_loss_by_epoch"].values())
    history["loss_finite"] = bool(loss_values) and all(math.isfinite(value) for value in loss_values)
    history["loss_decreased"] = (
        len(loss_values) >= 2 and float(loss_values[-1]) < float(loss_values[0])
    )
    return history


def plot_training_curves(run_dir: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.read_csv(metrics_csv_path(run_dir))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    loss_column = "train/loss_epoch" if "train/loss_epoch" in frame.columns else "train/loss"
    if loss_column in frame.columns:
        values = frame[["epoch", loss_column]].dropna().groupby("epoch")[loss_column].last()
        axes[0].plot(values.index + 1, values.values, marker="o", label=loss_column)
    if "val/loss" in frame.columns:
        values = frame[["epoch", "val/loss"]].dropna().groupby("epoch")["val/loss"].last()
        axes[0].plot(values.index + 1, values.values, marker="o", label="val/loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for column in ("val/mAP_50_95", "val/ema_mAP_50_95", "val/mAP_50", "val/ema_mAP_50"):
        if column in frame.columns:
            values = frame[["epoch", column]].dropna().groupby("epoch")[column].last()
            axes[1].plot(values.index + 1, values.values, marker="o", label=column)
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def detections_to_predictions(detections: Any, classes: list[str]) -> list[dict[str, Any]]:
    if detections is None or detections.xyxy is None:
        return []
    xyxy = np.asarray(detections.xyxy)
    class_ids = (
        np.asarray(detections.class_id, dtype=int)
        if detections.class_id is not None
        else np.empty((0,), dtype=int)
    )
    scores = (
        np.asarray(detections.confidence, dtype=float)
        if detections.confidence is not None
        else np.ones((len(xyxy),), dtype=float)
    )
    output: list[dict[str, Any]] = []
    for box, class_id, score in zip(xyxy, class_ids, scores):
        if class_id < 0 or class_id >= len(classes):
            raise ValueError(f"Prediction contains invalid class id {class_id}")
        output.append(
            {
                "xyxy": [float(value) for value in box],
                "class_id": int(class_id),
                "class_name": classes[int(class_id)],
                "score": float(score),
            }
        )
    return output


def render_split_predictions(
    model: Any,
    dataset: dict[str, Any],
    role: str,
    output_dir: Path,
    threshold: float,
    limit: int | None,
) -> dict[str, Any]:
    classes = list(dataset["classes"])
    data = load_coco_split(dataset, role)
    _, category_to_index = category_mapping(data, classes, role)
    grouped = annotations_by_image(data)
    predicted_objects = 0
    rendered = 0
    image_rows = sorted(data["images"], key=lambda row: row["file_name"])
    if limit is not None:
        image_rows = image_rows[:limit]
    for image_info in image_rows:
        filename = str(image_info["file_name"])
        image_path = split_directory(dataset, role) / filename
        detections = model.predict(str(image_path), threshold=threshold)
        predictions = detections_to_predictions(detections, classes)
        ground_truth = [
            {
                "xyxy": xywh_to_xyxy(annotation["bbox"]),
                "class_id": category_to_index[int(annotation["category_id"])],
            }
            for annotation in grouped.get(int(image_info["id"]), [])
        ]
        draw_boxes(image_path, ground_truth, predictions, classes, output_dir / filename)
        rendered += 1
        predicted_objects += len(predictions)
    return {
        "rendered_images": rendered,
        "predicted_objects": predicted_objects,
        "threshold": threshold,
        "output_dir": str(output_dir),
    }


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def greedy_match(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            iou = box_iou(gt["xyxy"], pred["xyxy"])
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))
    candidates.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))

    return (
        matches,
        set(range(len(ground_truth))) - matched_gt,
        set(range(len(predictions))) - matched_pred,
    )


def save_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    csv_path: Path,
    png_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[int(value) for value in row]])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Test confusion matrix (IoU matching)")
    threshold = float(matrix.max()) / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=170)
    plt.close(fig)


def analyze_test_predictions(
    model: Any,
    dataset: dict[str, Any],
    output_dir: Path,
    confidence_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    role = "test"
    classes = list(dataset["classes"])
    data = load_coco_split(dataset, role)
    category_ids, category_to_index = category_mapping(data, classes, role)
    grouped = annotations_by_image(data)
    labels = classes + [BACKGROUND]
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    class_counts = {
        class_name: {"tp": 0, "fp": 0, "fn": 0} for class_name in classes
    }
    coco_predictions: list[dict[str, Any]] = []
    image_errors: list[dict[str, Any]] = []
    prediction_dir = output_dir / "predictions"
    error_dir = output_dir / "false_positive_false_negative"

    for image_info in sorted(data["images"], key=lambda row: row["file_name"]):
        image_id = int(image_info["id"])
        filename = str(image_info["file_name"])
        image_path = split_directory(dataset, role) / filename
        detections = model.predict(str(image_path), threshold=confidence_threshold)
        predictions = detections_to_predictions(detections, classes)
        ground_truth = [
            {
                "xyxy": xywh_to_xyxy(annotation["bbox"]),
                "class_id": category_to_index[int(annotation["category_id"])],
                "annotation_id": int(annotation["id"]),
            }
            for annotation in grouped.get(image_id, [])
        ]

        for prediction in predictions:
            x1, y1, x2, y2 = prediction["xyxy"]
            coco_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_ids[int(prediction["class_id"])],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(prediction["score"]),
                }
            )

        matches, unmatched_gt, unmatched_pred = greedy_match(
            ground_truth, predictions, iou_threshold
        )
        error_gt = set(unmatched_gt)
        error_pred = set(unmatched_pred)
        mismatches: list[dict[str, Any]] = []
        for gt_index, pred_index, iou in matches:
            gt_class = int(ground_truth[gt_index]["class_id"])
            pred_class = int(predictions[pred_index]["class_id"])
            matrix[gt_class, pred_class] += 1
            if gt_class == pred_class:
                class_counts[classes[gt_class]]["tp"] += 1
            else:
                class_counts[classes[gt_class]]["fn"] += 1
                class_counts[classes[pred_class]]["fp"] += 1
                error_gt.add(gt_index)
                error_pred.add(pred_index)
                mismatches.append(
                    {
                        "true": classes[gt_class],
                        "predicted": classes[pred_class],
                        "iou": iou,
                    }
                )

        for gt_index in unmatched_gt:
            class_id = int(ground_truth[gt_index]["class_id"])
            matrix[class_id, -1] += 1
            class_counts[classes[class_id]]["fn"] += 1
        for pred_index in unmatched_pred:
            class_id = int(predictions[pred_index]["class_id"])
            matrix[-1, class_id] += 1
            class_counts[classes[class_id]]["fp"] += 1

        draw_boxes(
            image_path,
            ground_truth,
            predictions,
            classes,
            prediction_dir / filename,
        )
        if error_gt or error_pred:
            draw_boxes(
                image_path,
                ground_truth,
                predictions,
                classes,
                error_dir / filename,
                error_gt=error_gt,
                error_pred=error_pred,
            )
            image_errors.append(
                {
                    "filename": filename,
                    "false_negatives": len(unmatched_gt),
                    "false_positives": len(unmatched_pred),
                    "class_mismatches": mismatches,
                }
            )

    for class_name, counts in class_counts.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        counts["precision"] = tp / (tp + fp) if tp + fp else 0.0
        counts["recall"] = tp / (tp + fn) if tp + fn else 0.0

    write_json(output_dir / "test_predictions.coco.json", coco_predictions)
    write_json(output_dir / "error_analysis.json", image_errors)
    write_json(output_dir / "threshold_metrics_by_class.json", class_counts)
    save_confusion_matrix(
        matrix,
        labels,
        output_dir / "confusion_matrix.csv",
        output_dir / "confusion_matrix.png",
    )
    return {
        "confidence_threshold": confidence_threshold,
        "match_iou_threshold": iou_threshold,
        "images": len(data["images"]),
        "predictions": len(coco_predictions),
        "images_with_errors": len(image_errors),
        "class_counts_at_threshold": class_counts,
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
    }


def structured_test_metrics(metrics: dict[str, float], classes: list[str]) -> dict[str, Any]:
    return {
        "overall": {
            "mAP50": metrics.get("test/mAP_50"),
            "mAP50_95": metrics.get("test/mAP_50_95"),
            "precision": metrics.get("test/precision"),
            "recall": metrics.get("test/recall"),
            "F1": metrics.get("test/F1"),
            "mAR": metrics.get("test/mAR"),
        },
        "AP_by_class": {
            class_name: metrics.get(f"test/AP/{class_name}") for class_name in classes
        },
        "raw": metrics,
    }


def preview_limit(value: Any) -> int | None:
    if value is None or str(value).lower() == "all":
        return None
    limit = int(value)
    if limit <= 0:
        raise ValueError("smoke.preview_images must be positive or 'all'.")
    return limit


def model_runs_root(config: dict[str, Any]) -> Path:
    return (
        Path(config["runs_root"])
        / str(config["dataset"]["name"])
        / str(config["model"]["name"])
    )


def finalize_smoke_run(config: dict[str, Any], run_dir: Path) -> Path:
    """Finish smoke audit/previews from an already trained checkpoint."""
    dataset = config["dataset"]
    classes = list(dataset["classes"])
    report_path = run_dir / "preflight_report.json"
    report = read_json(report_path) if report_path.is_file() else preflight(config)
    limit = preview_limit(config.get("smoke", {}).get("preview_images", "all"))
    best_checkpoint = run_dir / "checkpoint_best_total.pth"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Training produced no best checkpoint: {best_checkpoint}")

    audit = checkpoint_audit(best_checkpoint, classes)
    write_json(run_dir / "checkpoint_audit.json", audit)
    history = metric_history(run_dir)
    write_json(run_dir / "metrics_history.json", history)
    plot_training_curves(run_dir, run_dir / "artifacts" / "training_curves.png")

    best_model = load_best_model(best_checkpoint, config["model"]["name"], classes)
    preview = render_split_predictions(
        best_model,
        dataset,
        "valid",
        run_dir / "artifacts" / "valid_predictions",
        float(config.get("evaluation", {}).get("confidence_threshold", 0.25)),
        limit,
    )
    release_model(best_model)

    automatic_checks = {
        "dataset_lock": report["dataset_lock"]["status"] == "PASS",
        "class_mapping": audit["saved_classes"] == classes,
        "head_class_count": audit["num_classes"] == len(classes),
        "cuda": report["environment"]["cuda_available"],
        "checkpoint": best_checkpoint.is_file(),
        "loss_finite": history["loss_finite"],
        "loss_decreased": history["loss_decreased"],
        "prediction_previews": preview["rendered_images"] > 0,
        "test_split_unused": True,
    }
    gate = {
        "status": "PASS" if all(automatic_checks.values()) else "FAIL",
        "checked_at": now_iso(),
        "automatic_checks": automatic_checks,
        "manual_review_required": True,
        "review_directory": preview["output_dir"],
        "best_checkpoint": str(best_checkpoint),
        "best_checkpoint_source": audit["best_total_source"],
        "loss": {
            "column": history["loss_column"],
            "by_epoch": history["train_loss_by_epoch"],
        },
        "prediction_preview": preview,
    }
    write_json(run_dir / "smoke_gate.json", gate)
    update_run_manifest(
        run_dir,
        status="SMOKE_PASS_PENDING_REVIEW" if gate["status"] == "PASS" else "SMOKE_FAIL",
        finished_at=now_iso(),
        smoke_gate=gate["status"],
        best_checkpoint=str(best_checkpoint),
        test_split_used=False,
        recovered_after_postprocess_failure=(run_dir / "failure.json").is_file(),
    )
    print(f"SMOKE_RUN={run_dir}")
    print(f"SMOKE_GATE={gate['status']}")
    print(f"REVIEW={preview['output_dir']}")
    return run_dir


def run_smoke(config: dict[str, Any]) -> Path:
    report = preflight(config)
    run_dir = create_run_dir(config, "smoke")
    dataset = config["dataset"]
    classes = list(dataset["classes"])
    phase_config = config.get("smoke", {})
    limit = preview_limit(phase_config.get("preview_images", "all"))
    rfdetr_dataset_dir = prepare_rfdetr_dataset_view(dataset, run_dir)
    kwargs = train_kwargs(config, "smoke", run_dir, rfdetr_dataset_dir)
    update_run_manifest(
        run_dir,
        status="RUNNING",
        phase="smoke",
        started_at=now_iso(),
        dataset_name=dataset["name"],
        dataset_dir=dataset["root"],
        rfdetr_dataset_dir=str(rfdetr_dataset_dir),
        train_config=config["_config_path"],
        dataset_config=dataset["_config_path"],
        model_name=config["model"]["name"],
        classes=classes,
        test_split_used=False,
    )
    write_json(run_dir / "preflight_report.json", report)
    write_json(run_dir / "resolved_config.json", config)
    write_json(run_dir / "train_kwargs.json", kwargs)

    try:
        save_ground_truth_previews(
            dataset,
            "train",
            run_dir / "artifacts" / "train_ground_truth",
            limit,
        )
        save_ground_truth_previews(
            dataset,
            "valid",
            run_dir / "artifacts" / "valid_ground_truth",
            limit,
        )

        random.seed(int(config["training"]["seed"]))
        np.random.seed(int(config["training"]["seed"]))
        cls = model_class(config["model"]["name"])
        model = cls(pretrain_weights=config["model"]["pretrained_checkpoint"])
        model.train(**kwargs)
        release_model(model)
        model = None
        return finalize_smoke_run(config, run_dir)
    except Exception as exc:
        write_json(
            run_dir / "failure.json",
            {"failed_at": now_iso(), "error": str(exc), "traceback": traceback.format_exc()},
        )
        update_run_manifest(run_dir, status="FAILED", failed_at=now_iso(), test_split_used=False)
        raise


def approve_smoke(run_dir: Path, acknowledged: bool) -> Path:
    if not acknowledged:
        raise ValueError("Approval requires --yes-i-checked after inspecting every preview image.")
    gate_path = run_dir / "smoke_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"Missing smoke gate: {gate_path}")
    gate = read_json(gate_path)
    if gate.get("status") != "PASS":
        raise ValueError("Cannot approve a smoke run whose automatic gate did not PASS.")
    preview_dir = Path(gate["review_directory"])
    if not preview_dir.is_dir() or not any(preview_dir.glob("*.jpg")):
        raise FileNotFoundError(f"Smoke prediction previews are missing: {preview_dir}")
    approval_path = run_dir / "smoke_approved.json"
    write_json(
        approval_path,
        {
            "approved": True,
            "approved_at": now_iso(),
            "acknowledgement": (
                "Human inspected smoke prediction previews for box position and color/class mapping."
            ),
            "review_directory": str(preview_dir),
        },
    )
    update_run_manifest(run_dir, status="SMOKE_APPROVED", approved_at=now_iso())
    return approval_path


def find_latest_approved_smoke(runs_dir: Path) -> Path:
    candidates = sorted(runs_dir.glob("smoke_*"), reverse=True)
    for candidate in candidates:
        gate_path = candidate / "smoke_gate.json"
        approval_path = candidate / "smoke_approved.json"
        if gate_path.is_file() and approval_path.is_file():
            if read_json(gate_path).get("status") == "PASS" and read_json(approval_path).get(
                "approved"
            ):
                return candidate
    raise FileNotFoundError(
        f"No approved smoke run found under {runs_dir}. Run smoke, inspect previews, then approve-smoke."
    )


def validate_smoke_for_train(config: dict[str, Any], smoke_run: Path | None) -> Path:
    run_dir = smoke_run or find_latest_approved_smoke(model_runs_root(config))
    gate_path = run_dir / "smoke_gate.json"
    approval_path = run_dir / "smoke_approved.json"
    if not gate_path.is_file() or read_json(gate_path).get("status") != "PASS":
        raise ValueError(f"Smoke run has not passed automatic checks: {run_dir}")
    if not approval_path.is_file() or read_json(approval_path).get("approved") is not True:
        raise ValueError(f"Smoke run has not been manually approved: {run_dir}")
    smoke_config = read_json(run_dir / "resolved_config.json")
    if smoke_config["dataset"] != config["dataset"]:
        raise ValueError("Dataset config changed after smoke approval; run and approve smoke again.")
    if smoke_config["model"] != config["model"]:
        raise ValueError("Model config changed after smoke approval; run and approve smoke again.")
    if smoke_config["training"] != config["training"]:
        raise ValueError("Training config changed after smoke approval; run and approve smoke again.")
    smoke_preflight_path = run_dir / "preflight_report.json"
    if not smoke_preflight_path.is_file():
        raise ValueError("Smoke run has no preflight report; run and approve smoke again.")
    smoke_lock_sha = (
        read_json(smoke_preflight_path).get("dataset_lock", {}).get("lock_sha256")
    )
    current_lock_path = (
        Path(config["dataset"]["root"]) / str(config["dataset"]["lock_file"])
    )
    current_lock_sha = sha256_file(current_lock_path)
    if smoke_lock_sha != current_lock_sha:
        raise ValueError(
            "Dataset lock changed after smoke approval; run and approve smoke again."
        )
    return run_dir.resolve()


def run_train(config: dict[str, Any], smoke_run: Path | None) -> Path:
    approved_smoke = validate_smoke_for_train(config, smoke_run)
    report = preflight(config)
    run_dir = create_run_dir(config, "train")
    dataset = config["dataset"]
    classes = list(dataset["classes"])
    dataset_dir = Path(dataset["root"])
    rfdetr_dataset_dir = prepare_rfdetr_dataset_view(dataset, run_dir)
    kwargs = train_kwargs(config, "train", run_dir, rfdetr_dataset_dir)
    update_run_manifest(
        run_dir,
        status="RUNNING",
        phase="train",
        started_at=now_iso(),
        dataset_name=dataset["name"],
        dataset_dir=str(dataset_dir),
        rfdetr_dataset_dir=str(rfdetr_dataset_dir),
        train_config=config["_config_path"],
        dataset_config=dataset["_config_path"],
        model_name=config["model"]["name"],
        classes=classes,
        approved_smoke_run=str(approved_smoke),
        test_split_used=False,
        test_evaluation_count=0,
    )
    write_json(run_dir / "preflight_report.json", report)
    write_json(run_dir / "resolved_config.json", config)
    write_json(run_dir / "train_kwargs.json", kwargs)

    try:
        cls = model_class(config["model"]["name"])
        model = cls(pretrain_weights=config["model"]["pretrained_checkpoint"])
        model.train(**kwargs)

        best_checkpoint = run_dir / "checkpoint_best_total.pth"
        if not best_checkpoint.is_file():
            raise FileNotFoundError(f"Training produced no best checkpoint: {best_checkpoint}")
        audit = checkpoint_audit(best_checkpoint, classes)
        write_json(run_dir / "checkpoint_audit.json", audit)
        history = metric_history(run_dir)
        write_json(run_dir / "metrics_history.json", history)
        plot_training_curves(run_dir, run_dir / "artifacts" / "training_curves.png")

        # Test is deliberately evaluated exactly once, after reloading the overall best
        # validation checkpoint selected across regular and EMA weights.
        test_metrics_path = run_dir / "test_results" / "test_metrics.json"
        if test_metrics_path.exists():
            raise RuntimeError("Refusing to evaluate test twice in the same run.")
        release_model(model)
        model = None
        best_model = load_best_model(best_checkpoint, config["model"]["name"], classes)
        evaluation_kwargs = {
            "dataset_dir": str(rfdetr_dataset_dir),
            "output_dir": str(run_dir / "test_results" / "_eval_runtime"),
            "dataset_file": "roboflow",
            "class_names": classes,
            "device": config["training"].get("device", "cuda"),
            "batch_size": int(config["training"].get("batch_size", 4)),
            "num_workers": int(config["training"].get("num_workers", 2)),
            "log_per_class_metrics": True,
            "progress_bar": config["training"].get("progress_bar", "tqdm"),
        }
        raw_metrics = best_model.evaluate(split="test", **evaluation_kwargs)
        test_metrics = structured_test_metrics(raw_metrics, classes)
        test_metrics.update(
            {
                "evaluated_at": now_iso(),
                "checkpoint": str(best_checkpoint),
                "checkpoint_sha256": audit["sha256"],
                "checkpoint_source": audit["best_total_source"],
                "test_images": report["dataset"]["splits"]["test"]["images"],
                "test_annotations": report["dataset"]["splits"]["test"]["annotations"],
            }
        )
        write_json(test_metrics_path, test_metrics)
        update_run_manifest(
            run_dir,
            status="ANALYZING_TEST_ERRORS",
            test_split_used=True,
            test_evaluation_count=1,
            test_evaluated_at=now_iso(),
        )

        analysis = analyze_test_predictions(
            best_model,
            dataset,
            run_dir / "test_results",
            float(config.get("evaluation", {}).get("confidence_threshold", 0.25)),
            float(config.get("evaluation", {}).get("match_iou_threshold", 0.50)),
        )
        write_json(run_dir / "test_results" / "analysis_summary.json", analysis)
        final_report = {
            "status": "PASS",
            "finished_at": now_iso(),
            "best_checkpoint": str(best_checkpoint),
            "best_checkpoint_source": audit["best_total_source"],
            "test_metrics": test_metrics,
            "error_analysis": analysis,
            "important_artifacts": {
                "training_curves": str(run_dir / "artifacts" / "training_curves.png"),
                "metrics_csv": str(metrics_csv_path(run_dir)),
                "confusion_matrix": str(
                    run_dir / "test_results" / "confusion_matrix.png"
                ),
                "false_positive_false_negative_images": str(
                    run_dir / "test_results" / "false_positive_false_negative"
                ),
            },
        }
        write_json(run_dir / "final_report.json", final_report)
        update_run_manifest(
            run_dir,
            status="COMPLETE",
            finished_at=now_iso(),
            best_checkpoint=str(best_checkpoint),
            best_checkpoint_source=audit["best_total_source"],
            test_split_used=True,
            test_evaluation_count=1,
        )
        print(f"TRAIN_RUN={run_dir}")
        print(f"BEST_CHECKPOINT={best_checkpoint}")
        print(f"TEST_RESULTS={run_dir / 'test_results'}")
        return run_dir
    except Exception as exc:
        write_json(
            run_dir / "failure.json",
            {"failed_at": now_iso(), "error": str(exc), "traceback": traceback.format_exc()},
        )
        manifest_path = run_dir / "run_manifest.json"
        manifest = read_json(manifest_path)
        update_run_manifest(
            run_dir,
            status="FAILED",
            failed_at=now_iso(),
            test_split_used=manifest.get("test_split_used", False),
            test_evaluation_count=manifest.get("test_evaluation_count", 0),
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reusable RF-DETR COCO smoke/train/evaluation pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config (default: {DEFAULT_CONFIG})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="Verify dataset lock, COCO mapping, and GPU.")
    subparsers.add_parser("smoke", help="Run the two-epoch smoke test.")

    finalize = subparsers.add_parser(
        "finalize-smoke",
        help="Re-run checkpoint audit and previews for an existing smoke run.",
    )
    finalize.add_argument("--run-dir", type=Path, required=True)

    approve = subparsers.add_parser(
        "approve-smoke", help="Confirm manual review of smoke prediction previews."
    )
    approve.add_argument("--run-dir", type=Path, required=True)
    approve.add_argument("--yes-i-checked", action="store_true")

    train = subparsers.add_parser(
        "train", help="Run configured training and final test evaluation."
    )
    train.add_argument(
        "--smoke-run",
        type=Path,
        help="Approved smoke run. Defaults to the latest approved smoke run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
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
    elif args.command == "smoke":
        run_smoke(config)
    elif args.command == "finalize-smoke":
        finalize_smoke_run(config, args.run_dir.resolve())
    elif args.command == "approve-smoke":
        approval = approve_smoke(args.run_dir.resolve(), args.yes_i_checked)
        print(f"SMOKE_APPROVAL={approval}")
    elif args.command == "train":
        smoke_run = args.smoke_run.resolve() if args.smoke_run else None
        run_train(config, smoke_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
