"""Write detailed CSV and summary JSON for SAM3 human review."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def bbox_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, aw, ah = (float(value) for value in first)
    bx1, by1, bw, bh = (float(value) for value in second)
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _duration_seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return round(
        (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(),
        3,
    )


def _write_labeling_report(
    path: Path, reviewed_items: list[dict[str, Any]]
) -> None:
    fields = [
        "file_name",
        "proposal_id",
        "action",
        "class_name",
        "proposal_score",
        "proposal_bbox",
        "final_bbox",
        "proposal_final_bbox_iou",
        "reviewed_at",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in reviewed_items:
            row = dict(item)
            for field in ("proposal_bbox", "final_bbox"):
                value = row.get(field)
                row[field] = json.dumps(value, ensure_ascii=False) if value else ""
            writer.writerow(row)


def write_review_report(
    config: dict[str, Any], review: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = Path(config["output"]["proposals_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decisions = [
        {**decision, "file_name": image["file_name"]}
        for image in review["images"]
        for decision in image["decisions"]
    ]
    counts = Counter(decision["action"] for decision in decisions)
    proposal_total = sum(
        1 for decision in decisions if decision["proposal_id"] is not None
    )
    reviewed_items = []
    for decision in decisions:
        iou = None
        if decision["proposal_bbox"] and decision["final_bbox"]:
            iou = bbox_iou(decision["proposal_bbox"], decision["final_bbox"])
        reviewed_items.append(
            {
                "file_name": decision["file_name"],
                "proposal_id": decision["proposal_id"],
                "action": decision["action"],
                "class_name": decision["class_name"],
                "proposal_score": decision.get("proposal_score"),
                "proposal_bbox": decision.get("proposal_bbox"),
                "final_bbox": decision.get("final_bbox"),
                "proposal_final_bbox_iou": iou,
                "reviewed_at": decision.get("reviewed_at"),
            }
        )

    def rate(action: str) -> float:
        return round(counts[action] / proposal_total, 6) if proposal_total else 0.0

    summary = {
        "schema_version": 1,
        "status": "PASS" if review["status"] == "complete" else "INCOMPLETE",
        "proposal_is_ground_truth": False,
        "proposal_processing": {
            "started_at": manifest.get("started_at"),
            "completed_at": manifest.get("completed_at"),
            "duration_seconds": _duration_seconds(
                manifest.get("started_at"), manifest.get("completed_at")
            ),
        },
        "human_review": {
            "started_at": review.get("started_at"),
            "completed_at": review.get("completed_at"),
            "duration_seconds": _duration_seconds(
                review.get("started_at"), review.get("completed_at")
            ),
            "images_total": len(review["images"]),
            "images_reviewed": sum(
                bool(image.get("reviewed")) for image in review["images"]
            ),
        },
        "summary": {
            "proposals_reviewed": proposal_total,
            "accepted": counts["accept"],
            "edited": counts["edit"],
            "rejected": counts["reject"],
            "missed": counts["missed"],
            "accept_rate": rate("accept"),
            "edit_rate": rate("edit"),
            "reject_rate": rate("reject"),
            "missed_per_reviewed_proposal": (
                round(counts["missed"] / proposal_total, 6)
                if proposal_total
                else float(counts["missed"])
            ),
        },
    }
    _write_labeling_report(
        Path(config["output"]["labeling_report"]), reviewed_items
    )
    summary_path = Path(config["output"]["summary_file"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
