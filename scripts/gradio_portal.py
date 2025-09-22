"""Gradio portal for running MolmoAct checkpoints interactively.

Launch the UI after installing dependencies:

```
pip install gradio  # included in the "serve" extra as well
python scripts/gradio_portal.py --checkpoint <hf_or_local_path> [--device cuda]
```
"""

from __future__ import annotations

import argparse
import itertools
import math
from typing import Any, List, Sequence

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

import gradio as gr


def _load_images(files: Sequence[Any]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for file in files or []:
        if file is None:
            continue
        if hasattr(file, "name") and isinstance(file.name, str):
            path = file.name
        else:
            path = str(file)
        if not path:
            continue
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def _max_images_from_config(model: AutoModelForImageTextToText) -> int:
    mm_cfg = getattr(model.config, "mm_preprocessor", None)
    limit = getattr(mm_cfg, "max_images", None)
    if isinstance(limit, int) and limit > 0:
        return limit
    return 1


def _cast_for_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = tensor.to(device)
    if torch.is_floating_point(tensor):
        # Use bfloat16 when available to mirror the evaluation script.
        if device.type == "cuda":
            tensor = tensor.to(torch.bfloat16)
    return tensor


def _normalize_trace(points: Sequence[Sequence[float]], width: int, height: int) -> List[tuple[float, float]]:
    if not points:
        return []
    xs = [p[0] for p in points if len(p) >= 2]
    ys = [p[1] for p in points if len(p) >= 2]
    if not xs or not ys:
        return []
    if all(0.0 <= x <= 1.0 for x in xs) and all(0.0 <= y <= 1.0 for y in ys):
        scaled = [(float(x) * width, float(y) * height) for x, y in zip(xs, ys)]
    else:
        scaled = [(float(x), float(y)) for x, y in zip(xs, ys)]
    return scaled


def _draw_traces_on_images(images: Sequence[Image.Image], traces: Sequence[Sequence[Sequence[float]]]) -> List[Image.Image]:
    if not images:
        return []
    if not traces:
        return [img.copy() for img in images]

    color_cycle = itertools.cycle(["#FF0054", "#0099FF", "#7CFFB2", "#FFB800", "#B388FF"])
    overlays: List[Image.Image] = []
    traces_per_image = max(1, math.ceil(len(traces) / len(images)))

    for img_idx, img in enumerate(images):
        canvas = img.copy().convert("RGB")
        draw = ImageDraw.Draw(canvas)
        start = img_idx * traces_per_image
        end = min(len(traces), start + traces_per_image)
        for trace_idx in range(start, end):
            color = next(color_cycle)
            points = _normalize_trace(traces[trace_idx], *canvas.size)
            if len(points) < 2:
                continue
            draw.line(points, fill=color, width=4, joint="curve")
            for x, y in points:
                radius = 6
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        overlays.append(canvas)
    return overlays


def build_portal(checkpoint: str, device: str) -> gr.Blocks:
    torch_device = torch.device(device)
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        torch_dtype="auto",
        padding_side="left",
    )
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map={"": str(torch_device)} if torch_device.type == "cuda" else None,
    ).to(torch_device)
    model.eval()

    max_images = _max_images_from_config(model)
    generation_config = GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>")

    def infer(
        user_message: str,
        uploaded_files: list[Any],
        chat_history: list[tuple[str, str]],
        message_state: list[dict[str, Any]],
    ) -> tuple[
        list[tuple[str, str]],
        list[dict[str, Any]],
        Any,
        Any,
        Any,
        list[Any],
    ]:
        if message_state is None:
            message_state = []
        if chat_history is None:
            chat_history = []

        user_message = (user_message or "").strip()
        if not user_message:
            gr.Warning("Please enter an instruction for the assistant.")
            return chat_history, message_state, None, None, None, []

        images = _load_images(uploaded_files)
        if not images:
            gr.Warning("Please upload at least one image before submitting.")
            return chat_history, message_state, None, None, None, []

        if len(images) > max_images:
            gr.Warning(f"Received {len(images)} images, but the model only supports {max_images}. Extra images are ignored.")
            images = images[:max_images]

        message_state = list(message_state)
        message_state.append({"role": "user", "content": [{"type": "text", "text": user_message}]})

        prompt = processor.apply_chat_template(message_state, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            images=[images],
            text=prompt,
            padding=True,
            return_tensors="pt",
        )

        tensor_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                tensor_inputs[key] = _cast_for_device(value, torch_device)
            else:
                tensor_inputs[key] = value

        with torch.inference_mode():
            if torch_device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    generated = model.generate(
                        **tensor_inputs,
                        generation_config=generation_config,
                        tokenizer=processor.tokenizer,
                    )
            else:
                generated = model.generate(
                    **tensor_inputs,
                    generation_config=generation_config,
                    tokenizer=processor.tokenizer,
                )

        prompt_length = tensor_inputs["input_ids"].shape[1]
        generated_tokens = generated[:, prompt_length:]
        generated_text = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        message_state.append({"role": "assistant", "content": [{"type": "text", "text": generated_text}]})
        chat_history = chat_history + [(user_message, generated_text)]

        try:
            depth_outputs = model.parse_depth(generated_text)
        except Exception as exc:  # noqa: BLE001
            depth_outputs = {"error": str(exc)}

        try:
            trace_outputs = model.parse_trace(generated_text)
        except Exception as exc:  # noqa: BLE001
            trace_outputs = {"error": str(exc)}

        try:
            action_outputs = model.parse_action(generated_text)
        except Exception as exc:  # noqa: BLE001
            action_outputs = {"error": str(exc)}

        overlays: list[Any]
        if isinstance(trace_outputs, Sequence) and not isinstance(trace_outputs, (dict, str)):
            overlays = _draw_traces_on_images(images, trace_outputs)
        else:
            overlays = [img.copy() for img in images]

        return (
            chat_history,
            message_state,
            depth_outputs,
            trace_outputs,
            action_outputs,
            overlays,
        )

    def reset_chat() -> tuple[list[tuple[str, str]], list[dict[str, Any]], None, None, None, list[Any]]:
        return [], [], None, None, None, []

    with gr.Blocks(title="MolmoAct Gradio Portal") as demo:
        gr.Markdown(
            f"""# MolmoAct Portal
Upload up to **{max_images}** image(s), provide instructions, and chat with the model.\n"
            "The portal parses predicted depth, traces, and actions from the response."""
        )

        with gr.Row():
            file_input = gr.File(label="Images", file_types=["image"], file_count="multiple")
            instruction = gr.Textbox(label="Instruction", placeholder="Describe the task for the robot...", lines=4)

        chatbot = gr.Chatbot(label="Conversation", height=400)

        with gr.Row():
            depth_json = gr.JSON(label="Depth Output")
            trace_json = gr.JSON(label="Trace Output")
            action_json = gr.JSON(label="Action Output")

        overlay_gallery = gr.Gallery(label="Trace Overlays", columns=2, height="auto", allow_preview=True)

        submit = gr.Button("Generate", variant="primary")
        clear = gr.Button("Clear Conversation")

        state_messages = gr.State([])

        submit.click(
            infer,
            inputs=[instruction, file_input, chatbot, state_messages],
            outputs=[chatbot, state_messages, depth_json, trace_json, action_json, overlay_gallery],
            show_progress=True,
        )
        clear.click(
            reset_chat,
            outputs=[chatbot, state_messages, depth_json, trace_json, action_json, overlay_gallery],
            queue=False,
        )

        demo.queue(concurrency_count=2)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the MolmoAct Gradio portal")
    parser.add_argument("--checkpoint", required=True, help="Local path or Hugging Face repo ID of the model checkpoint")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run inference on (default: cuda if available, else cpu)",
    )
    parser.add_argument("--share", action="store_true", help="Launch Gradio with public sharing enabled")
    args = parser.parse_args()

    demo = build_portal(args.checkpoint, args.device)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
