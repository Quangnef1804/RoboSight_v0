"""Preview and interactively review temporary SAM3 proposals."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _require_opencv_gui() -> None:
    build = cv2.getBuildInformation()
    gui_line = next(
        (line.strip() for line in build.splitlines() if line.strip().startswith("GUI:")),
        "",
    )
    if not gui_line or gui_line.upper().endswith("NONE"):
        raise RuntimeError(
            "OpenCV GUI is unavailable. Reinstall the Windows GUI wheel with "
            "`python -m pip install --force-reinstall --no-deps "
            "opencv-python==4.10.0.84`."
        )


def _screen_limits() -> tuple[int, int]:
    try:
        import ctypes

        user32 = ctypes.windll.user32
        return (
            max(640, round(user32.GetSystemMetrics(0) * 0.85)),
            max(480, round(user32.GetSystemMetrics(1) * 0.78)),
        )
    except Exception:
        return 1280, 800


def _fit_image(
    image: np.ndarray, max_width: int, max_height: int
) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale == 1.0:
        return image, scale
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _display_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    return _fit_image(image, *_screen_limits())


def _select_bbox(window: str, image: np.ndarray) -> list[float] | None:
    display, scale = _display_image(image)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, display.shape[1], display.shape[0])
    x, y, width, height = cv2.selectROI(
        window, display, showCrosshair=True, fromCenter=False
    )
    cv2.destroyWindow(window)
    if width <= 0 or height <= 0:
        return None
    return [
        float(x / scale),
        float(y / scale),
        float(width / scale),
        float(height / scale),
    ]


def _draw_proposal(
    image: np.ndarray, mask: np.ndarray, proposal: dict[str, Any]
) -> np.ndarray:
    rendered = image.copy()
    tint = np.zeros_like(rendered)
    tint[:, :] = (40, 210, 255)
    rendered[mask] = cv2.addWeighted(rendered, 0.45, tint, 0.55, 0)[mask]
    x, y, width, height = (round(value) for value in proposal["bbox"])
    cv2.rectangle(rendered, (x, y), (x + width, y + height), (0, 255, 255), 2)
    label = (
        f"#{proposal['id']} {proposal['suggested_class']} "
        f"{proposal['score']:.3f}"
    )
    for thickness, color in ((3, (0, 0, 0)), (1, (0, 255, 255))):
        cv2.putText(
            rendered,
            label,
            (max(0, x), max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return rendered


def _draw_controls(
    image: np.ndarray, selected_class: str, instructions: str
) -> np.ndarray:
    rendered = image.copy()
    panel_height = min(90, max(60, rendered.shape[0] // 12))
    overlay = rendered.copy()
    cv2.rectangle(
        overlay, (0, 0), (rendered.shape[1], panel_height), (18, 18, 18), -1
    )
    rendered = cv2.addWeighted(overlay, 0.82, rendered, 0.18, 0)
    lines = (f"Class: {selected_class}", instructions)
    for index, line in enumerate(lines):
        cv2.putText(
            rendered,
            line,
            (16, 30 + index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return rendered


def _draw_decisions(image: np.ndarray, decisions: list[dict[str, Any]]) -> np.ndarray:
    rendered = image.copy()
    colors = {
        "accept": (0, 220, 0),
        "edit": (0, 165, 255),
        "missed": (255, 0, 255),
    }
    for decision in decisions:
        bbox = decision.get("final_bbox")
        if not bbox:
            continue
        x, y, width, height = (round(value) for value in bbox)
        color = colors.get(decision["action"], (255, 255, 255))
        cv2.rectangle(rendered, (x, y), (x + width, y + height), color, 3)
        label = f"{decision['class_name']} ({decision['action']})"
        cv2.putText(
            rendered,
            label,
            (max(0, x), max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return rendered


def _class_from_key(key: int, classes: list[str]) -> str | None:
    if ord("1") <= key <= ord("9"):
        index = key - ord("1")
        if index < len(classes):
            return classes[index]
    return None


def _open_window(window: str, image: np.ndarray) -> None:
    display, _ = _display_image(image)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, display.shape[1], display.shape[0])


def _show(
    window: str,
    image: np.ndarray,
    selected_class: str,
    instructions: str,
) -> None:
    display, _ = _display_image(image)
    display = _draw_controls(display, selected_class, instructions)
    cv2.imshow(window, display)


def _window_closed(window: str) -> bool:
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def create_previews(
    config: dict[str, Any], manifest: dict[str, Any]
) -> list[Path]:
    """Save one proposal overview per input image."""
    proposal_dir = Path(config["output"]["proposals_dir"])
    preview_dir = Path(config["output"]["previews_dir"])
    preview_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for proposal_image in manifest["images"]:
        source = Path(proposal_image["source_path"])
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {source}")
        with np.load(proposal_dir / proposal_image["mask_file"]) as payload:
            masks = payload["masks"].astype(bool)
        rendered = image
        for proposal in proposal_image["proposals"]:
            rendered = _draw_proposal(
                rendered, masks[int(proposal["mask_index"])], proposal
            )
        preview_path = preview_dir / proposal_image["file_name"]
        if preview_path.exists() and not config["runtime"]["overwrite"]:
            raise FileExistsError(
                f"Preview already exists: {preview_path}. "
                "Set runtime.overwrite=true to replace it."
            )
        if not cv2.imwrite(str(preview_path), rendered):
            raise OSError(f"Cannot write preview: {preview_path}")
        generated.append(preview_path)
    return generated


def validate_preview_output(config: dict[str, Any]) -> None:
    preview_dir = Path(config["output"]["previews_dir"])
    if (
        preview_dir.exists()
        and any(preview_dir.iterdir())
        and not config["runtime"]["overwrite"]
    ):
        raise FileExistsError(
            f"Preview output is not empty: {preview_dir}. "
            "Set runtime.overwrite=true to replace generated previews."
        )


def _new_review(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "sam3_human_review",
        "status": "in_progress",
        "started_at": _now_iso(),
        "completed_at": None,
        "classes": config["dataset"]["classes"],
        "proposal_manifest": str(
            Path(config["output"]["proposals_dir"]) / "manifest.json"
        ),
        "images": [
            {
                "image_id": image["id"],
                "file_name": image["file_name"],
                "source_path": image["source_path"],
                "sha256": image["sha256"],
                "width": image["width"],
                "height": image["height"],
                "reviewed": False,
                "decisions": [],
            }
            for image in manifest["images"]
        ],
    }


def _review_one_image(
    proposal_image: dict[str, Any],
    review_image: dict[str, Any],
    classes: list[str],
    proposal_dir: Path,
) -> bool:
    source = Path(proposal_image["source_path"])
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {source}")
    with np.load(proposal_dir / proposal_image["mask_file"]) as payload:
        masks = payload["masks"].astype(bool)
    decided_ids = {
        decision["proposal_id"]
        for decision in review_image["decisions"]
        if decision.get("proposal_id") is not None
    }

    print(f"\nẢnh: {proposal_image['file_name']}")
    print("Class:", ", ".join(f"{i + 1}={name}" for i, name in enumerate(classes)))
    print("Phím review proposal: 1-5 chọn class | A accept | E edit | R reject | Q lưu/thoát")
    window = f"SAM3 review - {proposal_image['file_name']}"
    _open_window(window, image)
    selected_class = classes[0]
    for proposal in proposal_image["proposals"]:
        if proposal["id"] in decided_ids:
            continue
        selected_class = proposal["suggested_class"]
        while True:
            rendered = _draw_proposal(
                image, masks[int(proposal["mask_index"])], proposal
            )
            _show(
                window,
                rendered,
                selected_class,
                "1-5 class | A keep | E edit | R reject | Q quit",
            )
            key = cv2.waitKey(30) & 0xFF
            chosen_class = _class_from_key(key, classes)
            if chosen_class is not None:
                selected_class = chosen_class
                continue
            if key in {ord("a"), ord("A")}:
                review_image["decisions"].append(
                    {
                        "proposal_id": proposal["id"],
                        "action": "accept",
                        "class_name": selected_class,
                        "proposal_score": proposal["score"],
                        "proposal_bbox": proposal["bbox"],
                        "final_bbox": proposal["bbox"],
                        "reviewed_at": _now_iso(),
                    }
                )
                break
            if key in {ord("e"), ord("E")}:
                bbox = _select_bbox("Select corrected bounding box", image)
                if bbox is None:
                    print("Không chọn bbox; proposal vẫn đang chờ.")
                    continue
                review_image["decisions"].append(
                    {
                        "proposal_id": proposal["id"],
                        "action": "edit",
                        "class_name": selected_class,
                        "proposal_score": proposal["score"],
                        "proposal_bbox": proposal["bbox"],
                        "final_bbox": bbox,
                        "reviewed_at": _now_iso(),
                    }
                )
                break
            if key in {ord("r"), ord("R")}:
                review_image["decisions"].append(
                    {
                        "proposal_id": proposal["id"],
                        "action": "reject",
                        "class_name": None,
                        "proposal_score": proposal["score"],
                        "proposal_bbox": proposal["bbox"],
                        "final_bbox": None,
                        "reviewed_at": _now_iso(),
                    }
                )
                break
            if key in {ord("q"), ord("Q")} or _window_closed(window):
                cv2.destroyAllWindows()
                return False

    print("Phím hoàn tất ảnh: 1-5 chọn class | M thêm vật thể bị sót | D hoàn tất | Q lưu/thoát")
    while True:
        rendered = _draw_decisions(image, review_image["decisions"])
        _show(
            window,
            rendered,
            selected_class,
            "1-5 class | M missed | D done | Q quit",
        )
        key = cv2.waitKey(30) & 0xFF
        chosen_class = _class_from_key(key, classes)
        if chosen_class is not None:
            selected_class = chosen_class
            continue
        if key in {ord("m"), ord("M")}:
            bbox = _select_bbox("Select missed object", rendered)
            if bbox is None:
                print("Không chọn bbox.")
                continue
            review_image["decisions"].append(
                {
                    "proposal_id": None,
                    "action": "missed",
                    "class_name": selected_class,
                    "proposal_score": None,
                    "proposal_bbox": None,
                    "final_bbox": bbox,
                    "reviewed_at": _now_iso(),
                }
            )
            continue
        if key in {ord("d"), ord("D")}:
            review_image["reviewed"] = True
            review_image["completed_at"] = _now_iso()
            cv2.destroyAllWindows()
            return True
        if key in {ord("q"), ord("Q")} or _window_closed(window):
            cv2.destroyAllWindows()
            return False


def review_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Review proposals interactively and persist resumable decisions."""
    _require_opencv_gui()
    proposal_dir = Path(config["output"]["proposals_dir"])
    manifest_path = proposal_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Proposal manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    classes = config["dataset"]["classes"]
    if manifest.get("ground_truth") is not False:
        raise ValueError("Invalid proposal manifest: ground_truth must be false.")
    if manifest.get("classes") != classes:
        raise ValueError("Class list changed after proposals were generated.")

    review_path = Path(config["output"]["review_file"])
    if review_path.is_file() and not config["runtime"]["overwrite"]:
        review = _read_json(review_path)
    else:
        review = _new_review(manifest, config)
    if review.get("classes") != classes:
        raise ValueError("Class list changed after review started.")

    proposal_by_id = {image["id"]: image for image in manifest["images"]}
    for review_image in review["images"]:
        if review_image.get("reviewed"):
            continue
        proposal_image = proposal_by_id[review_image["image_id"]]
        completed = _review_one_image(
            proposal_image, review_image, classes, proposal_dir
        )
        _write_json(review_path, review)
        if not completed:
            print(f"Đã lưu tiến độ: {review_path}")
            return review

    review["status"] = "complete"
    review["completed_at"] = _now_iso()
    _write_json(review_path, review)
    return review
