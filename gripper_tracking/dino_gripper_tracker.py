"""Patch-level gripper tracking with DINOv3 features.

This module extracts DINOv3 patch embeddings for a sequence of frames and
tracks the gripper position in image coordinates by combining cosine similarity
with a simple motion prior. The tracker can be seeded with an initial patch
index and optional reference crop images (each with their own patch index).

The implementation is model-agnostic as long as the backbone behaves like a
ViT with class/register tokens followed by patch tokens (e.g. DINOv3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import AutoModel


def pad_to_multiple(pil_img: Image.Image, multiple: int) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Pad `pil_img` so height/width are multiples of `multiple`."""
    width, height = pil_img.size
    pad_w = int(math.ceil(width / multiple) * multiple - width)
    pad_h = int(math.ceil(height / multiple) * multiple - height)
    if pad_w == 0 and pad_h == 0:
        return pil_img, (0, 0, 0, 0)
    canvas = Image.new("RGB", (width + pad_w, height + pad_h), (0, 0, 0))
    canvas.paste(pil_img, (0, 0))
    return canvas, (0, 0, pad_w, pad_h)


def preprocess_image(pil_img: Image.Image, patch_size: int) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int, int, int]]:
    """Convert `pil_img` to a batched tensor and numpy display array."""
    padded_img, pad_box = pad_to_multiple(pil_img, patch_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(padded_img).unsqueeze(0)
    disp = np.array(padded_img, dtype=np.uint8)
    return tensor, disp, pad_box


@dataclass
class FramePatchGrid:
    """Patch embeddings computed for a frame."""

    embeddings: np.ndarray  # (N, D) un-normalized patch vectors
    normalized: np.ndarray  # (N, D) unit-norm patch vectors
    rows: int
    cols: int
    patch_size: int
    height: int  # original (unpadded) height
    width: int   # original (unpadded) width
    xs: np.ndarray  # (N,) patch centers in image coordinates (pixels)
    ys: np.ndarray  # (N,) patch centers in image coordinates (pixels)
    valid_mask: np.ndarray  # (N,) marks patches fully inside original image
    source_path: Optional[str] = None

    def idx_to_rc(self, idx: int) -> Tuple[int, int]:
        r = idx // self.cols
        c = idx % self.cols
        return r, c

    def idx_to_xy(self, idx: int) -> Tuple[float, float]:
        return float(self.xs[idx]), float(self.ys[idx])


@dataclass
class ReferencePatch:
    """Additional reference embedding sourced from a crop image."""

    patch_idx: int
    weight: float = 1.0
    image_path: Optional[str] = None
    image: Optional[Image.Image] = None
    start_frame: int = 0
    end_frame: Optional[int] = None  # inclusive; None means active forever
    description: Optional[str] = None

    def load_image(self) -> Image.Image:
        if self.image is not None:
            return self.image if self.image.mode == "RGB" else self.image.convert("RGB")
        if not self.image_path:
            raise ValueError("ReferencePatch requires either `image` or `image_path`.")
        with Image.open(self.image_path) as img:
            return img.convert("RGB")


@dataclass
class TrajectoryPoint:
    frame_idx: int
    patch_idx: int
    x: float
    y: float
    score: float
    similarity_prev: float
    similarity_refs: Dict[str, float] = field(default_factory=dict)
    smoothed_x: Optional[float] = None
    smoothed_y: Optional[float] = None


class DINOGripperTracker:
    """Track a gripper across frames using DINOv3 patch embeddings."""

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        device: Optional[torch.device] = None,
        patch_size_override: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = True,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=trust_remote_code)
        self.model.to(self.device)
        self.model.eval()

        config_patch = getattr(self.model.config, "patch_size", None)
        if isinstance(config_patch, (tuple, list)):
            # e.g. (16, 16)
            config_patch = int(config_patch[0])
        self.patch_size = patch_size_override or config_patch or 16

    @torch.inference_mode()
    def _encode_image(self, pil_img: Image.Image, source_path: Optional[str] = None) -> FramePatchGrid:
        tensor, _, _ = preprocess_image(pil_img, self.patch_size)
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        _, _, padded_h, padded_w = tensor.shape
        width, height = pil_img.size

        outputs = self.model(pixel_values=tensor)
        hidden = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()

        rows = padded_h // self.patch_size
        cols = padded_w // self.patch_size
        n_patches = rows * cols
        n_tokens = hidden.shape[0]
        n_special = n_tokens - n_patches
        if n_special < 1:
            raise RuntimeError(
                f"Unexpected token layout: tokens={n_tokens}, rows*cols={n_patches}, patch_size={self.patch_size}"
            )

        patches = hidden[n_special:, :]
        patches = patches.reshape(rows, cols, -1)
        embeddings = patches.reshape(n_patches, -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normalized = embeddings / norms

        grid_x = (np.arange(cols) + 0.5) * self.patch_size
        grid_y = (np.arange(rows) + 0.5) * self.patch_size
        xs, ys = np.meshgrid(grid_x, grid_y)
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)

        valid_mask = (xs < width) & (ys < height)

        return FramePatchGrid(
            embeddings=embeddings,
            normalized=normalized,
            rows=rows,
            cols=cols,
            patch_size=self.patch_size,
            height=height,
            width=width,
            xs=xs,
            ys=ys,
            valid_mask=valid_mask,
            source_path=source_path,
        )

    def render_patch_grid(
        self,
        image: Image.Image | str,
        highlight_idx: Optional[int] = None,
        annotate: bool = True,
    ) -> Image.Image:
        """Return an image with patch grid/indices overlaid for easier index lookup."""

        owns_image = False
        if isinstance(image, Image.Image):
            pil_img = image if image.mode == "RGB" else image.convert("RGB")
        else:
            pil_img = Image.open(image).convert("RGB")
            owns_image = True

        try:
            grid = self._encode_image(pil_img)
            overlay = self._draw_patch_grid(pil_img, grid, highlight_idx=highlight_idx, annotate=annotate)
        finally:
            if owns_image:
                pil_img.close()
        return overlay

    @staticmethod
    def _draw_patch_grid(
        pil_img: Image.Image,
        grid: FramePatchGrid,
        highlight_idx: Optional[int],
        annotate: bool,
    ) -> Image.Image:
        canvas = pil_img.copy()
        draw = ImageDraw.Draw(canvas)
        width, height = canvas.size
        ps = grid.patch_size

        # Grid lines (clamped to original width/height)
        for r in range(grid.rows + 1):
            y = r * ps
            if y > height:
                y = height
            draw.line((0, int(y), width, int(y)), fill=(255, 255, 0), width=1)
        for c in range(grid.cols + 1):
            x = c * ps
            if x > width:
                x = width
            draw.line((int(x), 0, int(x), height), fill=(255, 255, 0), width=1)

        # Highlight specific patch if requested
        if highlight_idx is not None:
            if 0 <= highlight_idx < grid.rows * grid.cols:
                r, c = grid.idx_to_rc(highlight_idx)
                x0 = max(0, c * ps)
                y0 = max(0, r * ps)
                x1 = min(width, (c + 1) * ps)
                y1 = min(height, (r + 1) * ps)
                draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=3)

        if annotate:
            font = ImageFont.load_default()
            for idx, valid in enumerate(grid.valid_mask):
                if not valid:
                    continue
                x = grid.xs[idx]
                y = grid.ys[idx]
                label = str(idx)
                try:
                    text_w, text_h = font.getsize(label)  # Pillow < 10
                except AttributeError:
                    if hasattr(font, "getbbox"):
                        bbox = font.getbbox(label)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                    elif hasattr(draw, "textbbox"):
                        bbox = draw.textbbox((0, 0), label, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                    else:
                        text_w, text_h = draw.textsize(label, font=font)
                x0 = int(max(0, x - text_w / 2))
                y0 = int(max(0, y - text_h / 2))
                draw.rectangle((x0 - 1, y0 - 1, x0 + text_w + 1, y0 + text_h + 1), fill=(0, 0, 0))
                draw.text((x0, y0), label, fill=(255, 255, 255), font=font)

        return canvas

    def encode_frames(self, frames: Sequence[Image.Image | str]) -> List[FramePatchGrid]:
        """Convert each frame to a patch grid."""
        grids: List[FramePatchGrid] = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                pil_img = frame if frame.mode == "RGB" else frame.convert("RGB")
                source_path = None
            else:
                with Image.open(frame) as img:
                    pil_img = img.convert("RGB")
                source_path = str(frame)
            grid = self._encode_image(pil_img, source_path=source_path)
            grids.append(grid)
            if not isinstance(frame, Image.Image):
                pil_img.close()
        return grids

    def _encode_reference(self, ref: ReferencePatch) -> Tuple[np.ndarray, str]:
        pil_img = ref.load_image()
        grid = self._encode_image(pil_img, source_path=ref.image_path)
        if ref.image is None:
            pil_img.close()
        if ref.patch_idx < 0 or ref.patch_idx >= grid.normalized.shape[0]:
            raise IndexError(
                f"Reference patch_idx {ref.patch_idx} out of range for image with {grid.normalized.shape[0]} patches"
            )
        embedding = grid.normalized[ref.patch_idx]
        label = ref.description or (ref.image_path or f"ref@{ref.patch_idx}")
        return embedding, label

    def track(
        self,
        frames: Sequence[Image.Image | str],
        initial_patch_idx: int,
        initial_frame_idx: int = 0,
        references: Optional[Iterable[ReferencePatch]] = None,
        weight_prev: float = 1.0,
        distance_penalty: float = 0.0,
        ema_alpha: Optional[float] = 0.3,
    ) -> List[TrajectoryPoint]:
        """Track the gripper patch across `frames`.

        Args:
            frames: Sequence of frame paths or PIL images.
            initial_patch_idx: Patch index (row-major) in the `initial_frame_idx` frame.
            initial_frame_idx: Frame whose patch index seeds the tracker (default: 0).
            references: Optional iterable of additional reference crop specs.
            weight_prev: Weight applied to similarity against the previous frame patch.
            distance_penalty: Multiplier applied to Euclidean distance penalty (pixels).
            ema_alpha: Optional decay for exponential smoothing in pixel space (None disables).
        """
        if initial_frame_idx != 0:
            raise NotImplementedError("Tracking currently expects initial_frame_idx=0.")

        grids = self.encode_frames(frames)
        if not grids:
            return []

        initial_grid = grids[initial_frame_idx]
        if initial_patch_idx < 0 or initial_patch_idx >= initial_grid.normalized.shape[0]:
            raise IndexError(
                f"initial_patch_idx {initial_patch_idx} out of range for frame with {initial_grid.normalized.shape[0]} patches"
            )

        ref_embeddings: List[Tuple[np.ndarray, ReferencePatch, str]] = []
        if references:
            for ref in references:
                embedding, label = self._encode_reference(ref)
                ref_embeddings.append((embedding, ref, label))

        trajectory: List[TrajectoryPoint] = []
        prev_idx = int(initial_patch_idx)
        prev_grid = initial_grid
        prev_embedding = prev_grid.normalized[prev_idx]
        prev_x, prev_y = prev_grid.idx_to_xy(prev_idx)
        smooth_x = prev_x
        smooth_y = prev_y

        labels = [label for _, _, label in ref_embeddings]
        initial_point = TrajectoryPoint(
            frame_idx=initial_frame_idx,
            patch_idx=prev_idx,
            x=prev_x,
            y=prev_y,
            score=weight_prev,
            similarity_prev=1.0,
            similarity_refs={label: 0.0 for label in labels},
            smoothed_x=smooth_x,
            smoothed_y=smooth_y,
        )
        trajectory.append(initial_point)

        for frame_idx in range(initial_frame_idx + 1, len(grids)):
            grid = grids[frame_idx]
            sims_prev = grid.normalized @ prev_embedding
            total_score = weight_prev * sims_prev
            ref_details: List[Tuple[str, float, np.ndarray]] = []

            for embedding, ref, label in ref_embeddings:
                if frame_idx < ref.start_frame:
                    continue
                if ref.end_frame is not None and frame_idx > ref.end_frame:
                    continue
                sims_ref = grid.normalized @ embedding
                total_score += ref.weight * sims_ref
                ref_details.append((label, ref.weight, sims_ref))

            if distance_penalty > 0.0:
                dx = grid.xs - prev_x
                dy = grid.ys - prev_y
                distances = np.sqrt(dx * dx + dy * dy)
                total_score -= distance_penalty * distances

            total_score = np.where(grid.valid_mask, total_score, -np.inf)
            if not np.any(np.isfinite(total_score)):
                raise RuntimeError(f"No valid patch candidates for frame {frame_idx}.")
            best_idx = int(np.argmax(total_score))
            best_score = float(total_score[best_idx])
            best_sim_prev = float(sims_prev[best_idx])

            prev_idx = best_idx
            prev_grid = grid
            prev_embedding = grid.normalized[best_idx]
            prev_x, prev_y = grid.idx_to_xy(best_idx)

            if ema_alpha is not None:
                smooth_x = ema_alpha * prev_x + (1.0 - ema_alpha) * smooth_x
                smooth_y = ema_alpha * prev_y + (1.0 - ema_alpha) * smooth_y
            else:
                smooth_x, smooth_y = prev_x, prev_y

            ref_scores = {label: float(sims_ref[best_idx]) for label, _, sims_ref in ref_details}

            point = TrajectoryPoint(
                frame_idx=frame_idx,
                patch_idx=best_idx,
                x=prev_x,
                y=prev_y,
                score=best_score,
                similarity_prev=best_sim_prev,
                similarity_refs=ref_scores,
                smoothed_x=smooth_x,
                smoothed_y=smooth_y,
            )
            trajectory.append(point)

        return trajectory


__all__ = [
    "DINOGripperTracker",
    "FramePatchGrid",
    "ReferencePatch",
    "TrajectoryPoint",
]
