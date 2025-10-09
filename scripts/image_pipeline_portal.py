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
    img = _to_uint8(image)
    if mask.ndim == 2:
        mask = mask[..., None]
    mask = mask.astype(bool)
    mask_3 = np.broadcast_to(mask, img.shape)
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


def _plot_patch_map(patch_idx: np.ndarray) -> np.ndarray:
    valid = patch_idx >= 0
    cmap = np.where(valid, patch_idx + 1, 0)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cmap, cmap="tab20")
    ax.set_title("Patch coverage map")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)


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
    }


def visualize_pipeline(
    image: np.ndarray,
    max_crops: int,
    overlap_margin: int,
    base_size: int,
    encoder: str,
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
    patch_map = _plot_patch_map(artifacts["patch_idx"])

    summary = (
        f"Tiling: {artifacts['tiling'][0]} x {artifacts['tiling'][1]}  |  "
        f"Crop count: {len(artifacts['crops'])}  |  "
        f"Canvas size: {artifacts['canvas'].shape[0]}x{artifacts['canvas'].shape[1]}"
    )

    return (
        original_img,
        canvas_img,
        canvas_overlay,
        global_img,
        global_overlay,
        crop_gallery,
        mask_gallery,
        patch_map,
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

        run_btn.click(
            fn=visualize_pipeline,
            inputs=[image_input, max_crops, overlap_margin, base_size, encoder_select],
            outputs=[
                original,
                canvas,
                canvas_mask,
                global_view,
                global_mask,
                crop_gallery,
                mask_gallery,
                patch_map,
                summary_box,
            ],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="MolmoAct image pipeline visualizer")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
