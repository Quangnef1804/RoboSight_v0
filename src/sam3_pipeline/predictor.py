"""Load pretrained SAM3 and create temporary, normalized mask proposals."""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .postprocess import filter_proposals


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def image_files(input_dir: Path, extensions: list[str]) -> list[Path]:
    supported = {extension.lower() for extension in extensions}
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported
    )


class Sam3Predictor:
    """Thin adapter around the checked-in SAM3 image API."""

    HUGGINGFACE_CHECKPOINTS = {
        "facebook/sam3": "sam3.pt",
        "facebook/sam3.1": "sam3.1_multiplex.pt",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = str(config["model"]["device"])
        self.threshold = float(config["postprocess"]["mask_threshold"])
        self.model = None
        self.processor = None
        self.checkpoint_descriptor = ""

    def _checkpoint_path(self) -> Path:
        model_config = self.config["model"]
        checkpoint_value = model_config.get("checkpoint")
        if checkpoint_value is not None:
            checkpoint = Path(checkpoint_value)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"SAM3 checkpoint not found: {checkpoint}. "
                    "Update model.checkpoint in configs/sam3.yaml."
                )
            self.checkpoint_descriptor = str(checkpoint)
            return checkpoint

        repo_id = str(model_config["repo_id"])
        checkpoint_name = self.HUGGINGFACE_CHECKPOINTS.get(repo_id)
        if checkpoint_name is None:
            raise ValueError(
                f"Unknown SAM3 checkpoint filename for Hugging Face repo {repo_id!r}. "
                "Set model.checkpoint to a local pretrained checkpoint."
            )
        try:
            if sys.platform == "win32":
                import truststore

                truststore.inject_into_ssl()
            from huggingface_hub import hf_hub_download

            checkpoint = Path(
                hf_hub_download(repo_id=repo_id, filename=checkpoint_name)
            )
        except Exception as error:
            raise RuntimeError(
                f"Cannot download pretrained SAM3 from {repo_id}. "
                "Confirm access and run `hf auth login`."
            ) from error
        self.checkpoint_descriptor = repo_id
        return checkpoint

    def load(self) -> None:
        source = Path(self.config["_repo_root"]) / "third-party" / "SAM3"
        if not source.is_dir():
            raise FileNotFoundError(f"SAM3 source directory not found: {source}")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ModuleNotFoundError as error:
            if error.name == "triton":
                package = "triton-windows" if sys.platform == "win32" else "triton"
                raise RuntimeError(
                    f"SAM3 requires {package}. Install project dependencies with "
                    "`pip install -r requirements.txt`."
                ) from error
            raise RuntimeError(
                "Cannot import SAM3. Install project dependencies with "
                "`pip install -r requirements.txt`."
            ) from error
        except Exception as error:
            raise RuntimeError(
                "Cannot import SAM3. Install project dependencies with "
                "`pip install -r requirements.txt`."
            ) from error

        checkpoint = self._checkpoint_path()
        self.model = build_sam3_image_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            device=self.device,
            eval_mode=True,
        )
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.threshold,
        )

    def _inference_context(self):
        if not self.device.lower().startswith("cuda"):
            return nullcontext()
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("model.device is CUDA but CUDA is not available.")
        precision = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        return torch.autocast(device_type="cuda", dtype=precision)

    def infer_image(self, image_path: Path) -> list[dict[str, Any]]:
        if self.processor is None:
            raise RuntimeError("Call load() before inference.")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            with self._inference_context():
                state = self.processor.set_image(image)
                proposals: list[dict[str, Any]] = []
                for class_name in self.config["dataset"]["classes"]:
                    output = self.processor.set_text_prompt(
                        prompt=str(class_name), state=state
                    )
                    masks = output["masks"].detach().cpu().numpy()
                    scores = (
                        output["scores"].detach().float().cpu().numpy().reshape(-1)
                    )
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks = masks[:, 0]
                    for mask, score in zip(masks, scores):
                        proposals.append(
                            {
                                "mask": np.asarray(mask, dtype=bool),
                                "score": float(score),
                                "suggested_class": str(class_name),
                            }
                        )
            return proposals


def propose_dataset(
    config: dict[str, Any], predictor: Sam3Predictor
) -> dict[str, Any]:
    """Run SAM3 and persist proposals separately from any ground truth."""
    input_dir = Path(config["input"]["images_dir"])
    proposal_dir = Path(config["output"]["proposals_dir"])
    manifest_path = proposal_dir / "manifest.json"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input image directory not found: {input_dir}")
    files = image_files(input_dir, config["input"]["extensions"])
    if not files:
        raise ValueError(f"No supported images found in: {input_dir}")
    if (
        proposal_dir.exists()
        and any(path.is_file() for path in proposal_dir.rglob("*"))
        and not config["runtime"]["overwrite"]
    ):
        raise FileExistsError(
            f"Proposal output is not empty: {proposal_dir}. "
            "Set runtime.overwrite=true to replace generated proposal files."
        )

    predictor.load()
    masks_dir = proposal_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    started_at = _now_iso()
    images: list[dict[str, Any]] = []
    proposal_id = 1

    for image_id, image_path in enumerate(files, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        raw = predictor.infer_image(image_path)
        postprocess = config["postprocess"]
        filtered = filter_proposals(
            raw,
            image_height=height,
            image_width=width,
            score_threshold=float(postprocess["mask_threshold"]),
            min_mask_area=int(postprocess["min_mask_area"]),
            duplicate_iou_threshold=float(postprocess["duplicate_iou_threshold"]),
            valid_region=list(postprocess["valid_region"]),
            valid_region_min_fraction=float(
                postprocess["valid_region_min_fraction"]
            ),
        )
        mask_name = f"{image_id:06d}.npz"
        np.savez_compressed(
            masks_dir / mask_name,
            masks=np.stack(
                [proposal["mask"] for proposal in filtered], axis=0
            )
            if filtered
            else np.empty((0, height, width), dtype=bool),
        )
        metadata = []
        for mask_index, proposal in enumerate(filtered):
            metadata.append(
                {
                    "id": proposal_id,
                    "score": proposal["score"],
                    "suggested_class": proposal["suggested_class"],
                    "bbox": proposal["bbox"],
                    "area": proposal["area"],
                    "valid_region_fraction": proposal["valid_region_fraction"],
                    "mask_index": mask_index,
                }
            )
            proposal_id += 1
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "source_path": str(image_path),
                "sha256": _sha256(image_path),
                "width": width,
                "height": height,
                "mask_file": f"masks/{mask_name}",
                "proposals": metadata,
            }
        )
        print(f"[{image_id}/{len(files)}] {image_path.name}: {len(metadata)} proposals")

    manifest = {
        "schema_version": 1,
        "kind": "sam3_proposals",
        "ground_truth": False,
        "status": "PROPOSAL_ONLY",
        "started_at": started_at,
        "completed_at": _now_iso(),
        "model_source": config["model"]["source"],
        "checkpoint": predictor.checkpoint_descriptor,
        "device": config["model"]["device"],
        "classes": config["dataset"]["classes"],
        "input_dir": str(input_dir),
        "images": images,
        "proposal_count": proposal_id - 1,
    }
    _write_json(manifest_path, manifest)
    return manifest
