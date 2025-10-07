#!/usr/bin/env python3
"""Interactive Gradio portal for browsing annotated episodes."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

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


def build_demo(data: Dict[str, Dict], chunk_size: int) -> gr.Blocks:
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

    def show_episode(episode_id: str):
        if not episode_id:
            empty_table = pd.DataFrame(columns=["Frame", "Milestone", "Reason"])
            return "Select an episode to view details.", empty_table
        entry = data.get(episode_id)
        if not entry:
            empty_table = pd.DataFrame(columns=["Frame", "Milestone", "Reason"])
            return f"Episode {episode_id} not found in annotations.", empty_table
        task = entry.get("task", "")
        picked = entry.get("picked", []) or []
        rows = []
        for item in picked:
            rows.append([
                item.get("frame_id", ""),
                item.get("milestone", ""),
                item.get("reason", ""),
            ])
        df = pd.DataFrame(rows, columns=["Frame", "Milestone", "Reason"])
        return f"**Task:** {task}\n\n**Episode:** {episode_id}", df

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

    def filtered_episode_ids(bounds: Tuple[int, int]) -> List[str]:
        low, high = int(bounds[0]), int(bounds[1])
        return [eid for eid in episode_ids if low <= lengths_lookup.get(eid, 0) <= high]

    def update_chunk(chunk_index: int, filtered_ids: List[str], ranges: List[Tuple[int, int]]):
        options = available_ids(filtered_ids, ranges, chunk_index)
        chunk_text = describe_chunk(ranges, chunk_index)
        if not options:
            placeholder_text, placeholder_table = show_episode("")
            return (
                gr.Radio.update(choices=[], value=None),
                chunk_text,
                placeholder_text,
                placeholder_table,
            )
        value = options[0]
        episode_text, episode_table = show_episode(value)
        return (
            gr.Radio.update(choices=options, value=value),
            chunk_text,
            episode_text,
            episode_table,
        )

    with gr.Blocks(title="Annotation Portal") as demo:
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
            filter_slider = gr.RangeSlider(
                minimum=min_length,
                maximum=max_length or 1,
                value=(min_length, max_length) if length_values else (0, 0),
                step=1,
                label="Filter episodes by milestone count",
            )
            distribution_summary_md = gr.Markdown(distribution_summary(initial_filtered_ids))

        distribution_plot = gr.Plot(value=render_distribution(initial_filtered_ids))

        with gr.Row():
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
        )

        task_markdown = gr.Markdown()
        frame_table = gr.Dataframe(headers=["Frame", "Milestone", "Reason"], interactive=False)

        filtered_ids_state = gr.State(initial_filtered_ids)
        chunk_ranges_state = gr.State(initial_ranges)

        chunk_slider.change(
            update_chunk,
            inputs=[chunk_slider, filtered_ids_state, chunk_ranges_state],
            outputs=[episode_selector, chunk_label, task_markdown, frame_table],
        )

        episode_selector.change(
            show_episode,
            inputs=episode_selector,
            outputs=[task_markdown, frame_table],
        )

        def apply_filter(bounds: Tuple[int, int]):
            filtered_ids = filtered_episode_ids(bounds)
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
                episode_text, episode_table = show_episode(value)
            else:
                value = None
                episode_text, episode_table = show_episode("")
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
            )

        filter_slider.change(
            apply_filter,
            inputs=filter_slider,
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
            ],
        )

        demo.load(
            show_episode,
            inputs=episode_selector,
            outputs=[task_markdown, frame_table],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch a Gradio portal for annotated episodes.")
    parser.add_argument("annotations", type=Path, help="Path to the JSON annotation file.")
    parser.add_argument("--chunk_size", type=int, default=40, help="Number of episode ids to show per chunk.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share mode if desired.")
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk_size must be positive")

    data = load_annotations(args.annotations)
    demo = build_demo(data, args.chunk_size)
    demo.queue().launch(share=args.share)


if __name__ == "__main__":
    main()
