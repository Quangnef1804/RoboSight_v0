"""Export only completed human-reviewed annotations to COCO."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _validate_review(
    review: dict[str, Any], manifest: dict[str, Any], classes: list[str]
) -> None:
    if review.get("status") != "complete":
        raise ValueError("Review is incomplete. Finish every image before export.")
    if review.get("classes") != classes or manifest.get("classes") != classes:
        raise ValueError("Class list does not match proposal/review metadata.")
    expected_ids = {
        proposal["id"]
        for image in manifest["images"]
        for proposal in image["proposals"]
    }
    decisions = [
        decision
        for image in review["images"]
        for decision in image.get("decisions", [])
    ]
    decided_ids = [
        decision["proposal_id"]
        for decision in decisions
        if decision.get("proposal_id") is not None
    ]
    if len(decided_ids) != len(set(decided_ids)) or set(decided_ids) != expected_ids:
        raise ValueError("Every SAM3 proposal must have exactly one review decision.")
    for decision in decisions:
        if decision["action"] != "reject" and decision.get("class_name") not in classes:
            raise ValueError(f"Invalid reviewed class: {decision.get('class_name')!r}")
        if decision["action"] != "reject" and not decision.get("final_bbox"):
            raise ValueError("A kept annotation is missing its final bounding box.")


def _checked_bbox(
    bbox: list[float], width: int, height: int, annotation_label: str
) -> list[float]:
    if len(bbox) != 4:
        raise ValueError(f"{annotation_label} bbox must have four values.")
    values = [float(value) for value in bbox]
    x, y, box_width, box_height = values
    if (
        not all(math.isfinite(value) for value in values)
        or x < 0
        or y < 0
        or box_width <= 0
        or box_height <= 0
        or x + box_width > width + 1e-6
        or y + box_height > height + 1e-6
    ):
        raise ValueError(f"{annotation_label} bbox is outside the image: {bbox}")
    return [round(value, 4) for value in values]


def export_coco(config: dict[str, Any]) -> Path:
    proposal_path = Path(config["output"]["proposals_dir"]) / "manifest.json"
    review_path = Path(config["output"]["review_file"])
    if not proposal_path.is_file():
        raise FileNotFoundError(f"Proposal manifest not found: {proposal_path}")
    if not review_path.is_file():
        raise FileNotFoundError(f"Review file not found: {review_path}")
    manifest = json.loads(proposal_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    classes = list(config["dataset"]["classes"])
    _validate_review(review, manifest, classes)

    images_dir = Path(config["input"]["images_dir"])
    dataset_root = images_dir.parent
    annotation_path = Path(config["output"]["annotation_file"])
    try:
        annotation_relative = annotation_path.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(
            "output.annotation_file must be inside the input dataset root."
        ) from error
    if annotation_path.exists() and not config["runtime"]["overwrite"]:
        raise FileExistsError(
            f"COCO annotation already exists: {annotation_path}. "
            "Set runtime.overwrite=true to replace it."
        )
    for candidate in (dataset_root, *dataset_root.parents):
        if (candidate / "dataset.lock.json").exists():
            raise PermissionError(f"Refusing to modify locked dataset: {candidate}")

    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    if dataset_manifest_path.is_file():
        current_manifest = json.loads(
            dataset_manifest_path.read_text(encoding="utf-8")
        )
        if current_manifest.get("kind") != "sam3_reviewed_dataset":
            raise PermissionError(f"Refusing to overwrite unrelated data: {dataset_root}")
        if not config["runtime"]["overwrite"]:
            raise FileExistsError(
                f"Dataset manifest already exists: {dataset_manifest_path}. "
                "Set runtime.overwrite=true to replace it."
            )

    review_by_id = {image["image_id"]: image for image in review["images"]}
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for source_image in manifest["images"]:
        source = Path(source_image["source_path"])
        if source.parent.resolve() != images_dir.resolve():
            raise ValueError(f"Proposal image is outside input.images_dir: {source}")
        if not source.is_file() or _sha256(source) != source_image["sha256"]:
            raise ValueError(
                f"Source image is missing or changed after proposal: {source}"
            )
        coco_images.append(
            {
                "id": source_image["id"],
                "file_name": source_image["file_name"],
                "width": source_image["width"],
                "height": source_image["height"],
            }
        )
        reviewed_image = review_by_id[source_image["id"]]
        for decision in reviewed_image["decisions"]:
            if decision["action"] == "reject":
                continue
            bbox = _checked_bbox(
                decision["final_bbox"],
                source_image["width"],
                source_image["height"],
                f"{source_image['file_name']} {decision['action']}",
            )
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": source_image["id"],
                    "category_id": classes.index(decision["class_name"]),
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 4),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1

    coco = {
        "info": {
            "description": dataset_root.name,
            "version": "1",
            "date_created": _now_iso(),
            "reviewed_ground_truth": True,
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": category_id, "name": name, "supercategory": "object"}
            for category_id, name in enumerate(classes)
        ],
    }
    _write_json(annotation_path, coco)
    _write_json(
        dataset_manifest_path,
        {
            "schema_version": 1,
            "kind": "sam3_reviewed_dataset",
            "name": dataset_root.name,
            "format": "coco",
            "created_at": _now_iso(),
            "image_dir": images_dir.relative_to(dataset_root).as_posix(),
            "annotation_file": annotation_relative.as_posix(),
            "classes": classes,
            "images": len(coco_images),
            "annotations": len(coco_annotations),
            "source_review": str(review_path),
            "proposal_is_ground_truth": False,
            "human_review_complete": True,
        },
    )
    return annotation_path
