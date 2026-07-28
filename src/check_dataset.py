"""Validate a reviewed, single-directory COCO dataset."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def validate_dataset(dataset: Path) -> dict[str, Any]:
    dataset = dataset.resolve()
    errors: list[str] = []
    manifest_path = dataset / "dataset_manifest.json"
    if not manifest_path.is_file():
        return {
            "status": "FAIL",
            "dataset": str(dataset),
            "images": 0,
            "annotations": 0,
            "categories": [],
            "objects_by_class": {},
            "errors": ["Missing dataset_manifest.json"],
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_dir = dataset / str(manifest.get("image_dir", "images"))
    annotation_path = dataset / str(
        manifest.get("annotation_file", "annotations/_annotations.coco.json")
    )
    if not annotation_path.is_file():
        return {
            "status": "FAIL",
            "dataset": str(dataset),
            "images": 0,
            "annotations": 0,
            "categories": [],
            "objects_by_class": {},
            "errors": [f"Missing COCO annotation: {annotation_path}"],
        }
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = coco.get("categories", [])
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    category_ids = [category.get("id") for category in categories]
    category_names = [category.get("name") for category in categories]
    if len(category_ids) != len(set(category_ids)):
        errors.append("Duplicate category ids.")
    if len(category_names) != len(set(category_names)):
        errors.append("Duplicate category names.")
    if category_names != manifest.get("classes"):
        errors.append("COCO categories do not match dataset manifest classes.")

    image_by_id: dict[Any, dict[str, Any]] = {}
    for image_record in images:
        image_id = image_record.get("id")
        if image_id in image_by_id:
            errors.append(f"Duplicate image id: {image_id}")
            continue
        image_by_id[image_id] = image_record
        path = image_dir / str(image_record.get("file_name", ""))
        if not path.is_file():
            errors.append(f"Missing image: {path}")
            continue
        try:
            with Image.open(path) as opened:
                actual_size = opened.size
            expected_size = (
                int(image_record.get("width", -1)),
                int(image_record.get("height", -1)),
            )
            if actual_size != expected_size:
                errors.append(
                    f"Image size mismatch for {path.name}: "
                    f"{actual_size} != {expected_size}"
                )
        except Exception as error:
            errors.append(f"Unreadable image {path}: {error}")

    annotation_ids: set[Any] = set()
    objects_by_class: Counter[str] = Counter()
    category_by_id = {
        category["id"]: category["name"]
        for category in categories
        if "id" in category and "name" in category
    }
    for annotation in annotations:
        annotation_id = annotation.get("id")
        if annotation_id in annotation_ids:
            errors.append(f"Duplicate annotation id: {annotation_id}")
        annotation_ids.add(annotation_id)
        image_record = image_by_id.get(annotation.get("image_id"))
        category_name = category_by_id.get(annotation.get("category_id"))
        if image_record is None:
            errors.append(f"Annotation {annotation_id} references an unknown image.")
            continue
        if category_name is None:
            errors.append(f"Annotation {annotation_id} references an unknown category.")
            continue
        bbox = annotation.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in bbox
            )
        ):
            errors.append(f"Annotation {annotation_id} has an invalid bbox.")
            continue
        x, y, width, height = (float(value) for value in bbox)
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > float(image_record["width"]) + 1e-6
            or y + height > float(image_record["height"]) + 1e-6
        ):
            errors.append(f"Annotation {annotation_id} bbox is outside its image.")
            continue
        objects_by_class[category_name] += 1

    if not manifest.get("human_review_complete"):
        errors.append("Dataset manifest does not confirm completed human review.")
    if manifest.get("proposal_is_ground_truth") is not False:
        errors.append("Dataset manifest must state proposal_is_ground_truth=false.")
    return {
        "status": "PASS" if not errors else "FAIL",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "images": len(images),
        "annotations": len(annotations),
        "categories": category_names,
        "objects_by_class": {
            name: objects_by_class.get(name, 0) for name in category_names
        },
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    report = validate_dataset(args.dataset)
    dataset_name = args.dataset.resolve().name
    report_path = (
        REPO_ROOT / "outputs" / "dataset_audit" / dataset_name / "validation.json"
    )
    _write_json(report_path, report)
    print(
        f"{report['status']}: {report['images']} images, "
        f"{report.get('annotations', 0)} annotations"
    )
    print(f"REPORT={report_path}")
    for error in report["errors"]:
        print(f"- {error}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
