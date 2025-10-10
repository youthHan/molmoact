"""Gradio portal visualizing MolmoAct image preprocessing."""
from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from olmo.hf_model.molmoact.image_processing_molmoact import (
    build_overlapping_crops,
    build_resized_image,
    resize_image,
    select_tiling,
)
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


@dataclass
class PipelineConfig:
    base_size: int = 378
    patch_size: int = 14
    resize_mode: str = "siglip"
    normalize_mode: str = "siglip"
    pad_value: float = 0.0
    max_crops: int = 8
    overlap_margin: int = 4

    @property
    def overlap_margins(self) -> Sequence[int]:
        return (self.overlap_margin, self.overlap_margin)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255).astype(np.uint8)
    return image


def _overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay a red tint on padded/invalid pixels.

    Accepts HxWx3 or 1xHxWx3 images; masks may be HxW, 1xHxW, HxWx1, or 1xHxWx1.
    """
    # Squeeze optional leading batch dim
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    img = _to_uint8(image)
    # Reduce mask to HxW boolean
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim == 2:
        mask = mask.astype(bool)
    else:
        mask = mask.astype(bool)
        # If mask already HxWxC, reduce via all-channels valid
        if mask.ndim == 3:
            mask = np.all(mask, axis=-1)
    # Broadcast to 3 channels
    mask_3 = np.broadcast_to(mask[..., None], img.shape)
    padded = ~mask_3
    tint = np.array([255, 0, 0], dtype=np.uint8)
    blended = (0.6 * img + 0.4 * tint).astype(np.uint8)
    result = np.where(padded, blended, img)
    return result


def _draw_boxes(canvas: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    palette = [
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (255, 215, 0),
        (186, 85, 211),
        (255, 140, 0),
    ]
    pil = Image.fromarray(_to_uint8(canvas))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except OSError:
        font = ImageFont.load_default()
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        color = palette[idx % len(palette)]
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), f"#{idx}", fill=color, font=font)
    return np.array(pil)


def _denormalize_image(image: np.ndarray, normalize_mode: str) -> np.ndarray:
    if normalize_mode == "siglip":
        return (image + 1.0) / 2.0
    if normalize_mode == "openai":
        std = np.array(OPENAI_CLIP_STD, dtype=np.float32)
        mean = np.array(OPENAI_CLIP_MEAN, dtype=np.float32)
        return image * std[None, None, :] + mean[None, None, :]
    if normalize_mode == "dino":
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        return image * std[None, None, :] + mean[None, None, :]
    return image


def _plot_patch_map(patch_idx: np.ndarray, show_ids: bool = False) -> np.ndarray:
    valid = patch_idx >= 0
    cmap = np.where(valid, patch_idx + 1, 0)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cmap, cmap="tab20")
    ax.set_title("Patch coverage map")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if show_ids:
        H, W = patch_idx.shape
        # only annotate if small grid to avoid clutter
        if H * W <= 18 * 18:
            for r in range(H):
                for c in range(W):
                    if valid[r, c]:
                        ax.text(c, r, str(int(patch_idx[r, c])), ha="center", va="center", fontsize=6, color="black")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)


def _colorize_ids(ids: np.ndarray, mode: str = "categorical") -> tuple[np.ndarray, np.ndarray]:
    """Colorize integer ids (HxW) with high contrast.

    - mode="categorical": map ids modulo a fixed bright palette (stable, vivid)
    - mode="continuous": use a matplotlib colormap over the id rank (pastel)
    Returns (rgb_image, valid_mask)
    """
    ids = ids.astype(np.int64)
    valid = ids >= 0

    if mode == "continuous":
        import matplotlib.cm as cm
        uniq = np.unique(ids[valid]) if np.any(valid) else np.array([0])
        if uniq.size == 0:
            uniq = np.array([0])
        lut = {v: i for i, v in enumerate(uniq.tolist())}
        norm = np.zeros_like(ids, dtype=np.float32)
        denom = max(1, len(uniq) - 1)
        for v in uniq:
            norm[ids == v] = lut[v] / denom
        mapper = cm.get_cmap("tab20")
        rgba = mapper(norm)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        return rgb, valid

    # Categorical: fixed bright palette
    palette = np.array([
        [31, 119, 180],  # blue
        [255, 127, 14],  # orange
        [44, 160, 44],   # green
        [214, 39, 40],   # red
        [148, 103, 189], # purple
        [140, 86, 75],   # brown
        [227, 119, 194], # pink
        [127, 127, 127], # gray
        [188, 189, 34],  # olive
        [23, 190, 207],  # cyan
        [255, 99, 71],   # tomato
        [60, 179, 113],  # mediumseagreen
        [65, 105, 225],  # royalblue
        [255, 215, 0],   # gold
        [186, 85, 211],  # mediumorchid
        [0, 191, 255],   # deepskyblue
    ], dtype=np.uint8)
    H, W = ids.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    idx = np.mod(ids, len(palette))
    rgb[valid] = palette[idx[valid]]
    return rgb, valid


def _upsample_patch_colors(colors_hw3: np.ndarray, patch_size: int) -> np.ndarray:
    """Nearest-neighbor upsample by repeating each patch to patch_size×patch_size pixels."""
    H, W, C = colors_hw3.shape
    return np.kron(colors_hw3, np.ones((patch_size, patch_size, 1), dtype=np.uint8))


def _overlay_alpha(base: np.ndarray, overlay: np.ndarray, alpha: float, mask: np.ndarray | None = None) -> np.ndarray:
    base_u8 = _to_uint8(base)
    ov_u8 = _to_uint8(overlay)
    if mask is None:
        return (alpha * ov_u8 + (1 - alpha) * base_u8).astype(np.uint8)
    # Ensure mask is HxW boolean, broadcast to 3 channels
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim == 2:
        mask3 = np.broadcast_to(mask[..., None], base_u8.shape)
    else:
        mask3 = np.broadcast_to(mask, base_u8.shape)
    blended = (alpha * ov_u8 + (1 - alpha) * base_u8).astype(np.uint8)
    return np.where(mask3, blended, base_u8)


def _draw_grid(image: np.ndarray, cell: int, color=(255, 255, 255), alpha: float = 0.5, thickness: int = 1) -> np.ndarray:
    """Overlay a grid with spacing `cell` pixels on top of image (no darkening between lines)."""
    img = _to_uint8(image)
    H, W = img.shape[:2]
    grid = np.zeros_like(img, dtype=np.uint8)
    # Draw horizontal lines
    for y in range(0, H, cell):
        y0 = max(0, y - thickness // 2)
        y1 = min(H, y0 + max(1, thickness))
        grid[y0:y1, :, :] = color
    # Draw vertical lines
    for x in range(0, W, cell):
        x0 = max(0, x - thickness // 2)
        x1 = min(W, x0 + max(1, thickness))
        grid[:, x0:x1, :] = color
    mask = (grid.sum(axis=-1) > 0)
    return _overlay_alpha(img, grid, alpha=alpha, mask=mask)


def _pool_cell_boxes_from_patch_idx(
    patch_idx: np.ndarray,
    patch_size: int,
    pool_h: int = 2,
    pool_w: int = 2,
) -> list[tuple[int, int, int, int]]:
    """Compute pooling cell boxes (in pixels) aligned with 2D pooling on the canvas.

    We mimic arange_for_pooling's symmetric padding by centering the patch grid to a
    multiple of (pool_h, pool_w), then enumerating each pooling window. A cell is kept
    if at least one patch in the window is valid (>=0).
    """
    H, W = patch_idx.shape
    H_pad = pool_h * ((H + pool_h - 1) // pool_h)
    W_pad = pool_w * ((W + pool_w - 1) // pool_w)
    pad_top = (H_pad - H) // 2
    pad_left = (W_pad - W) // 2

    boxes: list[tuple[int, int, int, int]] = []
    for r in range(0, H_pad, pool_h):
        for c in range(0, W_pad, pool_w):
            r0 = r - pad_top
            c0 = c - pad_left
            r1 = r0 + pool_h
            c1 = c0 + pool_w
            # Intersect with valid patch grid
            rr0, rr1 = max(0, r0), min(H, r1)
            cc0, cc1 = max(0, c0), min(W, c1)
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            sub = patch_idx[rr0:rr1, cc0:cc1]
            if not np.any(sub >= 0):
                continue
            y0 = max(0, r0) * patch_size
            x0 = max(0, c0) * patch_size
            y1 = min(H, r1) * patch_size
            x1 = min(W, c1) * patch_size
            boxes.append((x0, y0, x1, y1))
    return boxes


def _draw_boxes_overlay(image: np.ndarray, boxes: list[tuple[int, int, int, int]], color=(255, 255, 255), thickness: int = 2, alpha: float = 0.7) -> np.ndarray:
    """Draw rectangle outlines on top of image with alpha blending (no darkening elsewhere)."""
    base = _to_uint8(image)
    overlay = Image.fromarray(np.zeros_like(base, dtype=np.uint8))
    draw = ImageDraw.Draw(overlay)
    for (x0, y0, x1, y1) in boxes:
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=tuple(color), width=max(1, int(thickness)))
    overlay_np = np.array(overlay)
    mask = (overlay_np.sum(axis=-1) > 0)
    return _overlay_alpha(base, overlay_np, alpha=alpha, mask=mask)


def _rescale_patch_overlay_to_original(
    original: np.ndarray,
    canvas: np.ndarray,
    canvas_mask: np.ndarray,
    overlay_canvas: np.ndarray,
) -> np.ndarray:
    """Approximate overlay on the original image by removing pad from the canvas and resizing.

    Steps:
      1) Find the bounding box of True region in `canvas_mask`.
      2) Crop overlay_canvas to that ROI.
      3) Resize the overlay ROI to original HxW.
    """
    # Compute bounding box of valid pixels on canvas
    mask_bool = canvas_mask.astype(bool)
    rows = np.where(mask_bool.any(axis=1))[0]
    cols = np.where(mask_bool.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        # Fallback: return transparent overlay
        return np.zeros_like(original)
    y0, y1 = rows[0], rows[-1] + 1
    x0, x1 = cols[0], cols[-1] + 1
    roi = overlay_canvas[y0:y1, x0:x1]
    # Resize to original resolution
    pil = Image.fromarray(_to_uint8(roi))
    pil = pil.resize((original.shape[1], original.shape[0]), Image.BILINEAR)
    return np.array(pil).astype(np.uint8)


def extract_pipeline_artifacts(image: np.ndarray, cfg: PipelineConfig):
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0

    left_margin, right_margin = cfg.overlap_margins
    total_margin_pixels = cfg.patch_size * (left_margin + right_margin)
    crop_patches = cfg.base_size // cfg.patch_size
    crop_window_patches = crop_patches - (left_margin + right_margin)
    crop_window_size = crop_window_patches * cfg.patch_size

    tiling = select_tiling(
        image.shape[0] - total_margin_pixels,
        image.shape[1] - total_margin_pixels,
        crop_window_size,
        cfg.max_crops,
    )

    canvas_size = (
        tiling[0] * crop_window_size + total_margin_pixels,
        tiling[1] * crop_window_size + total_margin_pixels,
    )

    resized_canvas, canvas_mask = resize_image(
        image,
        cfg.resize_mode,
        canvas_size,
        cfg.pad_value,
    )

    global_resized, global_mask, _ = build_resized_image(
        image,
        cfg.resize_mode,
        cfg.normalize_mode,
        (cfg.base_size, cfg.base_size),
        cfg.pad_value,
        cfg.patch_size,
    )
    # Squeeze batch dim introduced by build_resized_image
    if global_resized.ndim == 4 and global_resized.shape[0] == 1:
        global_resized = global_resized[0]
    if global_mask.ndim == 3 and global_mask.shape[0] == 1:
        global_mask = global_mask[0]
    global_resized = _denormalize_image(global_resized, cfg.normalize_mode)
    global_mask = global_mask.astype(bool)

    crop_size = cfg.base_size
    crop_coords: List[Tuple[int, int, int, int]] = []
    crops: List[np.ndarray] = []
    crop_masks: List[np.ndarray] = []

    for i in range(tiling[0]):
        y0 = i * crop_window_size
        for j in range(tiling[1]):
            x0 = j * crop_window_size
            y1 = y0 + crop_size
            x1 = x0 + crop_size
            crop = resized_canvas[y0:y1, x0:x1]
            mask = canvas_mask[y0:y1, x0:x1].astype(bool)
            crops.append(crop)
            crop_masks.append(mask)
            crop_coords.append((x0, y0, x1, y1))

    _, _, patch_idx = build_overlapping_crops(
        image,
        cfg.resize_mode,
        cfg.normalize_mode,
        cfg.max_crops,
        list(cfg.overlap_margins),
        (cfg.base_size, cfg.base_size),
        cfg.pad_value,
        cfg.patch_size,
    )

    return {
        "original": image,
        "canvas": resized_canvas,
        "canvas_mask": canvas_mask.astype(bool),
        "global": global_resized,
        "global_mask": global_mask,
        "crops": crops,
        "crop_masks": crop_masks,
        "crop_boxes": crop_coords,
        "tiling": tiling,
        "patch_idx": patch_idx,
        "patch_size": cfg.patch_size,
    }


def visualize_pipeline(
    image: np.ndarray,
    max_crops: int,
    overlap_margin: int,
    base_size: int,
    encoder: str,
    show_grid: bool,
    show_ids: bool,
    grid_alpha: float,
    grid_thickness: int,
    show_pool_cells: bool,
    pool_alpha: float,
    pool_thickness: int,
):
    if image is None:
        raise gr.Error("Please upload an image to visualize.")

    encoder = encoder or "SigLIP2"
    encoder = encoder.lower()
    if "clip" in encoder:
        resize_mode = "default"
        normalize_mode = "openai"
    elif "dino" in encoder:
        resize_mode = "dino"
        normalize_mode = "dino"
    else:
        resize_mode = "siglip"
        normalize_mode = "siglip"

    cfg = PipelineConfig(
        max_crops=max_crops,
        overlap_margin=overlap_margin,
        base_size=base_size,
        resize_mode=resize_mode,
        normalize_mode=normalize_mode,
    )
    artifacts = extract_pipeline_artifacts(image, cfg)

    original_img = _to_uint8(artifacts["original"])
    canvas_img = _draw_boxes(artifacts["canvas"], artifacts["crop_boxes"])
    global_img = _to_uint8(artifacts["global"])
    global_overlay = _overlay_mask(artifacts["global"], artifacts["global_mask"])

    crop_gallery = []
    mask_gallery = []
    for idx, (crop, mask) in enumerate(zip(artifacts["crops"], artifacts["crop_masks"])):
        crop_gallery.append((_to_uint8(crop), f"Crop #{idx}"))
        mask_gallery.append((_overlay_mask(crop, mask), f"Crop #{idx} mask"))

    canvas_overlay = _overlay_mask(artifacts["canvas"], artifacts["canvas_mask"])
    patch_map = _plot_patch_map(artifacts["patch_idx"], show_ids=show_ids)

    summary = (
        f"Tiling: {artifacts['tiling'][0]} x {artifacts['tiling'][1]}  |  "
        f"Crop count: {len(artifacts['crops'])}  |  "
        f"Canvas size: {artifacts['canvas'].shape[0]}x{artifacts['canvas'].shape[1]}"
    )

    # Build overlays
    # Color mode is categorical for high contrast
    canvas_overlay_coverage = _overlay_alpha(
        artifacts["canvas"],
        _upsample_patch_colors(_colorize_ids(artifacts["patch_idx"], mode="categorical")[0], artifacts["patch_size"]),
        alpha=0.35,
        mask=artifacts["canvas_mask"],
    )
    original_overlay_coverage = _overlay_alpha(
        artifacts["original"],
        _rescale_patch_overlay_to_original(
            artifacts["original"],
            artifacts["canvas"],
            artifacts["canvas_mask"],
            _upsample_patch_colors(_colorize_ids(artifacts["patch_idx"], mode="categorical")[0], artifacts["patch_size"]),
        ),
        alpha=0.35,
        mask=np.ones(artifacts["original"].shape[:2], dtype=bool),
    )

    # Draw pooling cells first (thin, low alpha), then grid (optional), so we don't obscure colors
    if show_pool_cells:
        pool_boxes = _pool_cell_boxes_from_patch_idx(artifacts["patch_idx"], artifacts["patch_size"], 2, 2)
        canvas_overlay_coverage = _draw_boxes_overlay(
            canvas_overlay_coverage,
            pool_boxes,
            color=(0, 255, 255),
            thickness=int(pool_thickness),
            alpha=float(pool_alpha),
        )
        blank = np.zeros_like(artifacts["canvas"], dtype=np.uint8)
        boxes_canvas = _draw_boxes_overlay(
            blank,
            pool_boxes,
            color=(0, 255, 255),
            thickness=int(pool_thickness),
            alpha=1.0,
        )
        boxes_on_original = _rescale_patch_overlay_to_original(
            artifacts["original"], artifacts["canvas"], artifacts["canvas_mask"], boxes_canvas
        )
        original_overlay_coverage = _overlay_alpha(
            original_overlay_coverage, boxes_on_original, alpha=float(pool_alpha)
        )

    if show_grid:
        canvas_overlay_coverage = _draw_grid(
            canvas_overlay_coverage,
            artifacts["patch_size"],
            alpha=float(grid_alpha),
            thickness=int(grid_thickness),
        )
        # Approximate grid on original by resizing a grid from canvas
        grid_canvas = _draw_grid(
            np.zeros_like(artifacts["canvas"]) + 1.0,
            artifacts["patch_size"],
            alpha=1.0,
            thickness=int(grid_thickness),
        )
        grid_on_original = _rescale_patch_overlay_to_original(
            artifacts["original"],
            artifacts["canvas"],
            artifacts["canvas_mask"],
            grid_canvas,
        )
        original_overlay_coverage = _overlay_alpha(original_overlay_coverage, grid_on_original, alpha=float(grid_alpha))


    return (
        original_img,
        canvas_img,
        canvas_overlay,
        global_img,
        global_overlay,
        crop_gallery,
        mask_gallery,
        patch_map,
        canvas_overlay_coverage,
        original_overlay_coverage,
        summary,
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MolmoAct Image Pipeline Visualizer") as demo:
        gr.Markdown("""
        ## MolmoAct Image Preprocessing Visualizer
        Upload an image to inspect how MolmoAct tiles it into high-resolution crops, pads borders,
        and builds the patch-level masks used during encoding.
        """)
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Input image", type="numpy", image_mode="RGB")
                max_crops = gr.Slider(1, 12, value=8, step=1, label="Max crops")
                overlap_margin = gr.Slider(0, 8, value=4, step=1, label="Overlap margin (patches)")
                base_size = gr.Slider(224, 512, value=378, step=14, label="Crop size (pixels)")
                encoder_select = gr.Dropdown(
                    label="Vision encoder preset",
                    choices=["SigLIP2", "OpenAI CLIP", "DINOv2"],
                    value="SigLIP2",
                )
                with gr.Row():
                    show_grid = gr.Checkbox(value=False, label="Show patch grid")
                    show_ids = gr.Checkbox(value=False, label="Show patch ids (on map)")
                with gr.Row():
                    grid_alpha = gr.Slider(0.05, 0.6, value=0.2, step=0.05, label="Grid alpha")
                    grid_thickness = gr.Slider(1, 3, value=1, step=1, label="Grid thickness (px)")
                show_pool_cells = gr.Checkbox(value=False, label="Show pooling cells (2×2)")
                with gr.Row():
                    pool_alpha = gr.Slider(0.05, 0.6, value=0.25, step=0.05, label="Pooling cell alpha")
                    pool_thickness = gr.Slider(1, 3, value=1, step=1, label="Pooling cell thickness (px)")
                run_btn = gr.Button("Visualize", variant="primary")
            with gr.Column():
                summary_box = gr.Markdown("Ready.")
                original = gr.Image(label="Original")
                canvas = gr.Image(label="Resized canvas with crop boxes")
                canvas_mask = gr.Image(label="Canvas mask overlay")
        with gr.Row():
            global_view = gr.Image(label="Global low-res crop")
            global_mask = gr.Image(label="Global mask overlay")
        with gr.Row():
            crop_gallery = gr.Gallery(label="High-resolution crops", columns=3, preview=True)
            mask_gallery = gr.Gallery(label="Crop mask overlay", columns=3, preview=True)
        patch_map = gr.Image(label="Patch coverage map")
        with gr.Row():
            coverage_canvas = gr.Image(label="Patch coverage overlay (canvas)")
            coverage_original = gr.Image(label="Patch coverage overlay (original)")

        run_btn.click(
            fn=visualize_pipeline,
            inputs=[image_input, max_crops, overlap_margin, base_size, encoder_select, show_grid, show_ids, grid_alpha, grid_thickness, show_pool_cells, pool_alpha, pool_thickness],
            outputs=[
                original,
                canvas,
                canvas_mask,
                global_view,
                global_mask,
                crop_gallery,
                mask_gallery,
                patch_map,
                coverage_canvas,
                coverage_original,
                summary_box,
            ],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="MolmoAct image pipeline visualizer")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7870)
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
