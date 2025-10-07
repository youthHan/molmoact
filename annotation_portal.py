#!/usr/bin/env python3
"""Interactive Gradio portal for browsing annotated episodes."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt


def load_annotations(path: Path) -> Dict[str, Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Annotation file must contain a JSON object mapping episode ids to entries.")
    return data


def chunk_metadata(episode_ids: List[str], chunk_size: int) -> Tuple[int, List[Tuple[int, int]]]:
    total = len(episode_ids)
    if total == 0:
        return 0, []
    chunks = math.ceil(total / chunk_size)
    ranges: List[Tuple[int, int]] = []
    for idx in range(chunks):
        start = idx * chunk_size
        end = min(start + chunk_size, total)
        ranges.append((start, end))
    return chunks, ranges


def normalize_change_point(value: object) -> str:
    try:
        if value is None or value == "":
            return ""
        if isinstance(value, (int,)):
            return str(int(value))
        if isinstance(value, float):
            return str(int(value))
        text = str(value)
        if text.isdigit():
            return text
        return str(int(float(text)))
    except Exception:
        return str(value)


ManifestEntry = Tuple[str, Optional[Path]]

GALLERY_CSS = """
.frames-gallery img {
    object-fit: contain !important;
    width: 100% !important;
    height: auto !important;
}

.frames-gallery button {
    aspect-ratio: auto !important;
}
"""


def load_manifest_lookup(path: Optional[Path]) -> Dict[Tuple[str, str], ManifestEntry]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"episode_id", "change_point", "png"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest CSV is missing columns: {', '.join(sorted(missing))}")
    base_dir = path.parent
    lookup: Dict[Tuple[str, str], ManifestEntry] = {}
    for _, row in df.iterrows():
        ep_id = str(row["episode_id"])
        cp_key = normalize_change_point(row["change_point"])
        png_val = row["png"]
        if pd.isna(png_val):
            continue
        png_path = Path(str(png_val))
        base_for_entry: Optional[Path] = None
        if not png_path.is_absolute():
            base_for_entry = base_dir
        else:
            png_path = png_path.resolve()
        lookup[(ep_id, cp_key)] = (str(png_path), base_for_entry)
    return lookup


def build_demo(
    data: Dict[str, Dict],
    chunk_size: int,
    image_root: Optional[Path],
    image_pattern: Optional[str],
    manifest_lookup: Dict[Tuple[str, str], ManifestEntry],
) -> gr.Blocks:
    episode_ids = sorted(data.keys(), key=lambda x: (int(x) if x.isdigit() else x))
    lengths_lookup = {
        eid: len((data.get(eid, {}).get("picked") or []))
        for eid in episode_ids
    }
    total_chunks, chunk_ranges = chunk_metadata(episode_ids, chunk_size)

    def available_ids(filtered_ids: List[str], ranges: List[Tuple[int, int]], chunk_index: int) -> List[str]:
        if not ranges:
            return []
        idx = max(0, min(chunk_index, len(ranges) - 1))
        start, end = ranges[idx]
        return filtered_ids[start:end]

    def describe_chunk(ranges: List[Tuple[int, int]], chunk_index: int) -> str:
        if not ranges:
            return "No episodes match the current filter."
        chunk_index = max(0, min(chunk_index, len(ranges) - 1))
        start, end = ranges[chunk_index]
        return f"Chunk {chunk_index + 1}/{len(ranges)}: episodes {start + 1} – {end}"

    def resolve_frame_image(episode_id: str, frame_id: str) -> Optional[str]:
        manifest_key = (episode_id, normalize_change_point(frame_id))
        candidate_paths: List[Path] = []

        manifest_entry = manifest_lookup.get(manifest_key)
        if manifest_entry:
            path_str, entry_base = manifest_entry
            manifest_path = Path(path_str)
            if manifest_path.is_absolute():
                candidate_paths.append(manifest_path)
            else:
                if entry_base is not None:
                    candidate_paths.append((entry_base / manifest_path).resolve())
                if image_root is not None:
                    candidate_paths.append((image_root / manifest_path).resolve())
                candidate_paths.append(manifest_path.resolve(strict=False))
        else:
            if image_root is not None:
                pattern = image_pattern or "{image_root}/{episode_id}/{frame_id}.jpg"
                fmt = pattern.format(
                    image_root=str(image_root),
                    episode_id=episode_id,
                    frame_id=frame_id,
                )
                fallback_path = Path(fmt)
                if fallback_path.is_absolute():
                    candidate_paths.append(fallback_path.resolve())
                else:
                    candidate_paths.append((image_root / fallback_path).resolve())

        # Deduplicate paths while preserving order
        seen: set[str] = set()
        unique_paths: List[Path] = []
        for path in candidate_paths:
            key = str(path)
            if key not in seen:
                seen.add(key)
                unique_paths.append(path)

        for path in unique_paths:
            if path.exists():
                return str(path)

        return str(unique_paths[0]) if unique_paths else None

    def show_episode(episode_id: str):
        if not episode_id:
            empty_table = pd.DataFrame(columns=["Frame", "Milestone", "Reason"])
            return "Select an episode to view details.", empty_table, []
        entry = data.get(episode_id)
        if not entry:
            empty_table = pd.DataFrame(columns=["Frame", "Milestone", "Reason"])
            return f"Episode {episode_id} not found in annotations.", empty_table, []
        task = entry.get("task", "")
        picked = entry.get("picked", []) or []
        rows = []
        images: List[Tuple[str, str]] = []
        for item in picked:
            frame_id = str(item.get("frame_id", ""))
            milestone = item.get("milestone", "")
            reason = item.get("reason", "")
            rows.append([frame_id, milestone, reason])
            img_path = resolve_frame_image(episode_id, frame_id)
            if img_path:
                caption = f"Frame {frame_id}: {milestone}" if milestone else f"Frame {frame_id}"
                images.append((img_path, caption))
        df = pd.DataFrame(rows, columns=["Frame", "Milestone", "Reason"])
        return f"**Task:** {task}\n\n**Episode:** {episode_id}", df, images

    def lengths_for_ids(ids: List[str]) -> List[int]:
        return [lengths_lookup.get(eid, 0) for eid in ids]

    def render_distribution(ids: List[str]):
        fig, ax = plt.subplots(figsize=(5, 3))
        lengths = lengths_for_ids(ids)
        if lengths:
            ax.hist(lengths, bins=range(min(lengths), max(lengths) + 2), color="#4c72b0", edgecolor="black")
            ax.set_xlabel("Milestone count")
            ax.set_ylabel("Episode frequency")
            ax.set_title("Milestone length distribution")
        else:
            ax.text(0.5, 0.5, "No episodes", ha="center", va="center", fontsize=12)
            ax.axis("off")
        fig.tight_layout()
        return fig

    def distribution_summary(ids: List[str]) -> str:
        lengths = lengths_for_ids(ids)
        if not lengths:
            return "No episodes to summarize. Adjust the filter settings."
        total = len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        avg_len = sum(lengths) / total
        return (
            f"**Filtered episodes:** {total}  \
**Min milestones:** {min_len}  \
**Max milestones:** {max_len}  \
**Average milestones:** {avg_len:.2f}"
        )

    def filtered_episode_ids(low: Optional[float], high: Optional[float]) -> List[str]:
        if low is None:
            low = min_length
        if high is None:
            high = max_length
        low_i, high_i = int(low), int(high)
        if low_i > high_i:
            low_i, high_i = high_i, low_i
        return [eid for eid in episode_ids if low_i <= lengths_lookup.get(eid, 0) <= high_i]

    def update_chunk(chunk_index: int, filtered_ids: List[str], ranges: List[Tuple[int, int]]):
        options = available_ids(filtered_ids, ranges, chunk_index)
        chunk_text = describe_chunk(ranges, chunk_index)
        if not options:
            placeholder_text, placeholder_table, placeholder_gallery = show_episode("")
            return (
                gr.Radio.update(choices=[], value=None),
                chunk_text,
                placeholder_text,
                placeholder_table,
                placeholder_gallery,
            )
        value = options[0]
        episode_text, episode_table, gallery_items = show_episode(value)
        return (
            gr.Radio.update(choices=options, value=value),
            chunk_text,
            episode_text,
            episode_table,
            gallery_items,
        )

    with gr.Blocks(title="Annotation Portal", css=GALLERY_CSS) as demo:
        gr.Markdown("# Annotation Browser")
        gr.Markdown(
            "Use the filters to narrow episodes, browse chunks of ids, and click an id to inspect its annotated frames."
        )

        length_values = [lengths_lookup[eid] for eid in episode_ids]
        min_length = min(length_values) if length_values else 0
        max_length = max(length_values) if length_values else 0

        initial_chunk = 0
        initial_filtered_ids = episode_ids
        initial_ranges = chunk_ranges
        initial_ids = available_ids(initial_filtered_ids, initial_ranges, initial_chunk)
        initial_episode = initial_ids[0] if initial_ids else None

        with gr.Row():
            with gr.Column(scale=2):
                min_filter = gr.Number(
                    label="Min milestones (≥)",
                    value=min_length if length_values else None,
                    precision=0,
                )
            with gr.Column(scale=2):
                max_filter = gr.Number(
                    label="Max milestones (≤)",
                    value=max_length if length_values else None,
                    precision=0,
                )
            with gr.Column(scale=1):
                apply_filter_btn = gr.Button("Apply filter", variant="primary")

            with gr.Column(scale=3):
                distribution_plot = gr.Plot(value=render_distribution(initial_filtered_ids))
                distribution_summary_md = gr.Markdown(distribution_summary(initial_filtered_ids))

        with gr.Row():
            with gr.Column(scale=1, min_width=240):
                chunk_slider = gr.Slider(
                    minimum=0,
                    maximum=max(total_chunks - 1, 0),
                    step=1,
                    value=initial_chunk,
                    label="Chunk index",
                )
                chunk_label = gr.Markdown(describe_chunk(initial_ranges, initial_chunk))
                episode_selector = gr.Radio(
                    choices=initial_ids,
                    value=initial_episode,
                    label="Episode ids in chunk",
                    interactive=True,
                )
            with gr.Column(scale=3):
                task_markdown = gr.Markdown()
                frame_table = gr.Dataframe(headers=["Frame", "Milestone", "Reason"], interactive=False, wrap=True)
            with gr.Column(scale=3):
                image_gallery = gr.Gallery(label="Frames", show_label=True, columns=2, elem_classes=["frames-gallery"]) 

        filtered_ids_state = gr.State(initial_filtered_ids)
        chunk_ranges_state = gr.State(initial_ranges)

        chunk_slider.change(
            update_chunk,
            inputs=[chunk_slider, filtered_ids_state, chunk_ranges_state],
            outputs=[episode_selector, chunk_label, task_markdown, frame_table, image_gallery],
        )

        episode_selector.change(
            show_episode,
            inputs=episode_selector,
            outputs=[task_markdown, frame_table, image_gallery],
        )

        def apply_filter(min_bound: Optional[float], max_bound: Optional[float]):
            filtered_ids = filtered_episode_ids(min_bound, max_bound)
            new_total, new_ranges = chunk_metadata(filtered_ids, chunk_size)
            chunk_idx = 0
            options = available_ids(filtered_ids, new_ranges, chunk_idx)
            chunk_text = describe_chunk(new_ranges, chunk_idx)
            slider_update = gr.Slider.update(
                minimum=0,
                maximum=max(new_total - 1, 0),
                value=chunk_idx,
            )
            if options:
                value = options[0]
                episode_text, episode_table, gallery_items = show_episode(value)
            else:
                value = None
                episode_text, episode_table, gallery_items = show_episode("")
            radio_update = gr.Radio.update(choices=options, value=value)
            return (
                slider_update,
                chunk_text,
                radio_update,
                render_distribution(filtered_ids),
                distribution_summary(filtered_ids),
                filtered_ids,
                new_ranges,
                episode_text,
                episode_table,
                gallery_items,
            )

        apply_filter_btn.click(
            apply_filter,
            inputs=[min_filter, max_filter],
            outputs=[
                chunk_slider,
                chunk_label,
                episode_selector,
                distribution_plot,
                distribution_summary_md,
                filtered_ids_state,
                chunk_ranges_state,
                task_markdown,
                frame_table,
                image_gallery,
            ],
        )

        demo.load(
            show_episode,
            inputs=episode_selector,
            outputs=[task_markdown, frame_table, image_gallery],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch a Gradio portal for annotated episodes.")
    parser.add_argument("annotations", type=Path, help="Path to the JSON annotation file.")
    parser.add_argument("--chunk_size", type=int, default=40, help="Number of episode ids to show per chunk.")
    parser.add_argument(
        "--image_root",
        type=Path,
        default=None,
        help="Root directory containing episode frame images (optional).",
    )
    parser.add_argument(
        "--image_pattern",
        type=str,
        default=None,
        help="Optional Python format string for frame images. Available keys: image_root, episode_id, frame_id.",
    )
    parser.add_argument(
        "--manifest_csv",
        type=Path,
        default=None,
        help="Optional CSV manifest (triplets) mapping change_point indices to image paths.",
    )
    parser.add_argument("--share", action="store_true", help="Enable Gradio share mode if desired.")
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk_size must be positive")

    data = load_annotations(args.annotations)
    image_root = args.image_root.resolve() if args.image_root else None
    manifest_lookup = load_manifest_lookup(args.manifest_csv)

    allowed_paths: set[str] = set()
    if image_root is not None:
        allowed_paths.add(str(image_root))
    if args.manifest_csv is not None:
        allowed_paths.add(str(args.manifest_csv.parent.resolve()))
    for path_str, base_dir in manifest_lookup.values():
        try:
            image_path = Path(path_str)
            if image_path.is_absolute():
                allowed_paths.add(str(image_path.parent.resolve()))
            else:
                if base_dir is not None:
                    allowed_paths.add(str((base_dir / image_path).parent.resolve()))
                if image_root is not None:
                    allowed_paths.add(str((image_root / image_path).parent.resolve()))
        except Exception:
            continue

    demo = build_demo(data, args.chunk_size, image_root, args.image_pattern, manifest_lookup)

    launch_kwargs = {"share": args.share}
    if allowed_paths:
        launch_kwargs["allowed_paths"] = sorted(allowed_paths)

    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
