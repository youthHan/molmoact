"""Interactive portal to inspect MolmoAct parquet traces with end-effector overlays."""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:  # noqa: N818
    raise SystemExit("numpy is required for scripts/parquet_trace_portal.py") from exc

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:  # noqa: N818
    raise SystemExit("pyarrow is required for scripts/parquet_trace_portal.py") from exc

from PIL import Image, ImageDraw

try:
    import gradio as gr
except ModuleNotFoundError as exc:  # noqa: N818
    raise SystemExit("gradio is required for scripts/parquet_trace_portal.py") from exc


TRACE_NONE_LABEL = "(none)"


# ------------------------------
# Parquet helpers
# ------------------------------


@dataclass
class RowGroupIndex:
    row_group: int
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length


class ParquetShard:
    """Wrap a single parquet file and support random access into its rows."""

    def __init__(self, path: Path, start_offset: int) -> None:
        self.path = Path(path)
        self._file = pq.ParquetFile(self.path)
        self.start = start_offset
        self.rows = self._file.metadata.num_rows if self._file.metadata is not None else 0
        self.end = self.start + self.rows
        schema = self._file.schema_arrow
        self.columns = list(schema.names) if schema is not None else []

        self._row_groups: List[RowGroupIndex] = []
        cumulative = 0
        if self._file.metadata is not None:
            for rg_idx in range(self._file.metadata.num_row_groups):
                rg_meta = self._file.metadata.row_group(rg_idx)
                length = rg_meta.num_rows
                self._row_groups.append(RowGroupIndex(rg_idx, cumulative, length))
                cumulative += length
        else:
            self._row_groups.append(RowGroupIndex(0, 0, self.rows))

        self._cache_rg: Optional[int] = None
        self._cache_table = None

    def _row_group_for(self, local_index: int) -> Tuple[int, int]:
        if local_index < 0 or local_index >= self.rows:
            raise IndexError(f"Row {local_index} out of bounds for shard {self.path}")
        for info in self._row_groups:
            if info.start <= local_index < info.end:
                return info.row_group, local_index - info.start
        raise IndexError(f"Unable to locate row group for index {local_index} in {self.path}")

    def _read_row_group(self, rg_index: int):
        if self._cache_rg != rg_index:
            self._cache_table = self._file.read_row_group(rg_index)
            self._cache_rg = rg_index
        return self._cache_table

    def get_row(self, local_index: int) -> Dict[str, Any]:
        if self.rows == 0:
            raise IndexError(f"Shard {self.path} has no rows")
        rg_index, offset = self._row_group_for(local_index)
        table = self._read_row_group(rg_index)
        if table is None:
            raise RuntimeError(f"Failed to read row group {rg_index} from {self.path}")
        row = table.slice(offset, 1).to_pylist()
        if not row:
            raise RuntimeError(f"Row slice returned empty result for index {local_index} in {self.path}")
        return row[0]


class ParquetViewer:
    """Aggregate one or more parquet shards and expose dataset metadata."""

    def __init__(self, files: Sequence[Path]):
        if not files:
            raise ValueError("No parquet files provided")

        self.shards: List[ParquetShard] = []
        self.columns: List[str] = []
        self.total_rows = 0

        seen: set[str] = set()
        offset = 0
        for path in files:
            shard = ParquetShard(path, offset)
            if shard.rows == 0:
                offset = shard.end
                continue
            self.shards.append(shard)
            offset = shard.end
            self.total_rows += shard.rows
            for col in shard.columns:
                if col not in seen:
                    seen.add(col)
                    self.columns.append(col)

        if not self.shards or self.total_rows == 0:
            raise ValueError("The provided parquet files do not contain any rows")

    def __len__(self) -> int:
        return self.total_rows

    def get_row(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= self.total_rows:
            raise IndexError(f"Row {index} out of bounds (0 <= index < {self.total_rows})")
        for shard in self.shards:
            if shard.start <= index < shard.end:
                local_index = index - shard.start
                return shard.get_row(local_index)
        raise RuntimeError(f"Unable to locate shard for row {index}")

    def first_row(self) -> Optional[Dict[str, Any]]:
        if self.total_rows == 0:
            return None
        return self.get_row(0)


# ------------------------------
# Data parsing helpers
# ------------------------------


def _iter_parquet_paths(spec: str) -> List[Path]:
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Provide at least one parquet path or directory")

    matches: List[Path] = []
    for token in tokens:
        expanded = os.path.expanduser(token)
        path = Path(expanded)
        if path.is_dir():
            matches.extend(sorted(path.rglob("*.parquet")))
            continue
        if path.is_file() and path.suffix == ".parquet":
            matches.append(path)
            continue
        # Allow simple glob patterns
        globbed = list(Path().glob(expanded))
        for candidate in globbed:
            if candidate.is_file() and candidate.suffix == ".parquet":
                matches.append(candidate)

    unique = sorted({m.resolve() for m in matches})
    if not unique:
        raise FileNotFoundError(f"No parquet files found for input: {spec}")
    return unique


def _to_pil_image(value: Any) -> Image.Image:
    if value is None:
        raise ValueError("Image value is None")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str) and os.path.exists(value):
        return Image.open(value).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") and os.path.exists(value["path"]):
            return Image.open(value["path"]).convert("RGB")
        if value.get("array") is not None:
            return _to_pil_image(value["array"])
    if isinstance(value, (list, tuple)):
        arr = np.array(value)
        return _array_to_pil(arr)
    if isinstance(value, np.ndarray):
        return _array_to_pil(value)
    raise ValueError(f"Unsupported image container: {type(value)}")


def _array_to_pil(array: np.ndarray) -> Image.Image:
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected array with 3 dimensions, received shape {array.shape}")
    if array.dtype in (np.float32, np.float64):
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8)
    return Image.fromarray(array)


def _looks_like_image(value: Any) -> bool:
    try:
        _to_pil_image(value)
        return True
    except Exception:
        return False


def _coerce_point(item: Any) -> Optional[Tuple[float, float]]:
    if isinstance(item, dict):
        if {"x", "y"}.issubset(item.keys()):
            return float(item["x"]), float(item["y"])
        if {"u", "v"}.issubset(item.keys()):
            return float(item["u"]), float(item["v"])
        if "points" in item:
            return None
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        try:
            return float(item[0]), float(item[1])
        except (TypeError, ValueError):
            return None
    return None


def _flatten_traces(value: Any) -> List[List[Tuple[float, float]]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return []
        return _flatten_traces(parsed)
    if isinstance(value, dict):
        if "points" in value:
            return _flatten_traces(value["points"])
        traces: List[List[Tuple[float, float]]] = []
        for key in ("annotation", "trace", "trajectory", "positions", "points"):
            if key in value:
                traces.extend(_flatten_traces(value[key]))
        if traces:
            return traces
        for item in value.values():
            traces.extend(_flatten_traces(item))
        return traces
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        point = _coerce_point(value)
        if point is not None:
            return [[point]]
        if all(_coerce_point(item) is not None for item in value):
            return [[_coerce_point(item) for item in value if _coerce_point(item) is not None]]
        traces: List[List[Tuple[float, float]]] = []
        for item in value:
            traces.extend(_flatten_traces(item))
        return traces
    return []


def _normalize_trace(points: Sequence[Tuple[float, float]], width: int, height: int) -> List[Tuple[float, float]]:
    if not points:
        return []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if not xs or not ys:
        return []
    if all(0.0 <= x <= 1.0 for x in xs) and all(0.0 <= y <= 1.0 for y in ys):
        return [(float(x) * width, float(y) * height) for x, y in zip(xs, ys)]
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def _draw_traces(image: Image.Image, traces: Sequence[Sequence[Tuple[float, float]]]) -> Image.Image:
    if not traces:
        return image
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    colors = ["#ff0054", "#0099ff", "#7cffb2", "#ffb800", "#b388ff"]
    width, height = canvas.size
    for idx, raw_points in enumerate(traces):
        color = colors[idx % len(colors)]
        points = _normalize_trace(raw_points, width, height)
        if len(points) < 2:
            continue
        draw.line(points, fill=color, width=4, joint="curve")
        for x, y in points:
            radius = 5
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return canvas


def _summarize_traces(traces: Sequence[Sequence[Tuple[float, float]]]) -> str:
    if not traces:
        return "No trace available."
    lines = []
    for idx, trace in enumerate(traces, start=1):
        preview = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in trace[:8])
        more = "" if len(trace) <= 8 else f" ... ({len(trace)} points)"
        lines.append(f"Trace {idx}: {preview}{more}")
    return "\n".join(lines)


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes len={len(value)}>"
    if isinstance(value, Image.Image):
        return f"<PIL.Image size={value.size}>"
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_json(v) for v in value)
    return value


# ------------------------------
# Gradio UI wiring
# ------------------------------


def _load_dataset(spec: str) -> ParquetViewer:
    files = _iter_parquet_paths(spec)
    return ParquetViewer(files)


def build_interface(initial_path: Optional[str] = None) -> gr.Blocks:
    with gr.Blocks(title="MolmoAct Parquet Trace Viewer") as demo:
        gr.Markdown(
            "MolmoAct parquet viewer. Load a parquet file or directory to inspect rows and overlay end-effector traces."
        )

        viewer_state = gr.State(None)

        with gr.Row():
            dataset_input = gr.Textbox(
                label="Parquet path",
                placeholder="/path/to/dataset/*.parquet or /path/to/folder",
                value=initial_path or "",
            )
            load_button = gr.Button("Load dataset", variant="primary")

        status = gr.Markdown("No dataset loaded.")
        index_slider = gr.Slider(label="Row index", minimum=0, maximum=0, step=1, value=0, interactive=False)
        image_dropdown = gr.Dropdown(
            label="Image columns",
            choices=[],
            value=[],
            multiselect=True,
            interactive=False,
        )
        trace_dropdown = gr.Dropdown(
            label="Trace column",
            choices=[TRACE_NONE_LABEL],
            value=TRACE_NONE_LABEL,
            interactive=False,
        )

        gallery = gr.Gallery(label="Overlay", show_label=True)
        trace_text = gr.Markdown("Trace preview will appear here.")
        raw_json = gr.JSON(label="Raw row")

        def update_display(
            viewer: Optional[ParquetViewer],
            index: int,
            images: List[str],
            trace_col: Optional[str],
        ):
            if viewer is None:
                return [], "Load a dataset to view rows.", {}
            if len(viewer) == 0:
                return [], "Dataset is empty.", {}

            index = int(index)
            index = max(0, min(index, len(viewer) - 1))
            try:
                row = viewer.get_row(index)
            except Exception as exc:  # noqa: BLE001
                gr.Warning(f"Failed to read row {index}: {exc}")
                return [], f"Failed to read row {index}.", {}

            selected = images or []
            trace_key = trace_col if trace_col and trace_col != TRACE_NONE_LABEL else None
            overlay_traces: List[List[Tuple[float, float]]] = []
            if trace_key and trace_key in row:
                overlay_traces = _flatten_traces(row[trace_key])

            gallery_items = []
            for name in selected:
                if name not in row:
                    continue
                value = row[name]
                try:
                    pil_img = _to_pil_image(value)
                except Exception as exc:  # noqa: BLE001
                    gr.Warning(f"Column {name} is not a valid image: {exc}")
                    continue
                gallery_items.append((_draw_traces(pil_img, overlay_traces), name))

            trace_summary = _summarize_traces(overlay_traces) if overlay_traces else "No trace available."
            sanitized = {k: _sanitize_for_json(v) for k, v in row.items()}
            sanitized["__index__"] = index
            return gallery_items, trace_summary, sanitized

        def on_load(path: str):
            path = (path or "").strip()
            if not path:
                gr.Warning("Please provide a parquet path or directory.")
                return None, "Awaiting dataset path.", gr.update(), gr.update(), gr.update()
            try:
                viewer = _load_dataset(path)
            except Exception as exc:  # noqa: BLE001
                gr.Warning(f"Failed to load dataset: {exc}")
                return None, "Failed to load dataset.", gr.update(), gr.update(), gr.update()

            sample = viewer.first_row()
            image_candidates: List[str] = []
            if sample:
                for name, value in sample.items():
                    if _looks_like_image(value):
                        image_candidates.append(name)
            if not image_candidates:
                for fallback in ("image", "primary", "secondary", "wrist"):
                    if fallback in viewer.columns:
                        image_candidates.append(fallback)

            trace_candidates = [col for col in viewer.columns if "annot" in col.lower() or "trace" in col.lower() or "traj" in col.lower()]
            default_trace = trace_candidates[0] if trace_candidates else TRACE_NONE_LABEL
            trace_choices = [TRACE_NONE_LABEL] + viewer.columns

            status_md = (
                f"Loaded {len(viewer)} rows across {len(viewer.shards)} file(s).\n"
                f"Columns: {', '.join(viewer.columns)}"
            )

            return (
                viewer,
                status_md,
                gr.update(minimum=0, maximum=max(0, len(viewer) - 1), value=0, interactive=True),
                gr.update(choices=viewer.columns, value=image_candidates, interactive=True),
                gr.update(choices=trace_choices, value=default_trace, interactive=True),
            )

        load_button.click(
            fn=on_load,
            inputs=dataset_input,
            outputs=[viewer_state, status, index_slider, image_dropdown, trace_dropdown],
        ).then(
            fn=update_display,
            inputs=[viewer_state, index_slider, image_dropdown, trace_dropdown],
            outputs=[gallery, trace_text, raw_json],
        )

        dependencies = [viewer_state, index_slider, image_dropdown, trace_dropdown]
        for control in (index_slider, image_dropdown, trace_dropdown):
            control.change(
                fn=update_display,
                inputs=dependencies,
                outputs=[gallery, trace_text, raw_json],
            )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a parquet trace visualization portal.")
    parser.add_argument("--path", type=str, default=None, help="Parquet file, directory, or glob pattern to load on start.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio public sharing.")
    parser.add_argument("--server-port", type=int, default=None, help="Optional server port for Gradio.")
    parser.add_argument("--server-name", type=str, default=None, help="Optional server name for Gradio.")
    args = parser.parse_args()

    demo = build_interface(initial_path=args.path)
    demo.queue().launch(share=args.share, server_port=args.server_port, server_name=args.server_name)


if __name__ == "__main__":
    main()
