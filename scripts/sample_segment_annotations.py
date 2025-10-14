"""Sample tracking points around annotated frames in segmented trajectories.

Given:
  * Segment metadata produced by `segment_tracked_trajectory.py`
  * Per-segment tracking outputs from `track_segments.py`
  * Original parquet shards containing annotations

The script iterates through the annotation rows and, for each unique annotation
frame, returns up to `--max-samples` tracking points spanning from the
annotation frame (inclusive) to the end of the segment (inclusive). Samples are
uniformly spaced (always including start/end). Each record includes parquet
indices, local/global frame indices, segment offsets, and tracking coordinates.

Usage example::

    python3 MolmoAct/scripts/sample_segment_annotations.py \
        --segments-json run_01_segments.json \
        --tracks-dir segment_tracks \
        --parquet /mnt/data/libero/train-*.parquet \
        --annotation-column annotation \
        --max-samples 5 \
        --output annotation_samples.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional dependency for parquet scanning
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pa = None
    pq = None


@dataclass
class Segment:
    segment_idx: int
    start_index: int
    end_index: int
    length: int
    start_total_frame: int
    end_total_frame: int
    start_local_frame: int
    end_local_frame: int
    parquet_idx: Optional[int]
    task: Optional[str]


@dataclass
class TrackRecord:
    segment_idx: int
    segment_frame_idx: int
    global_frame_idx: int
    parquet_idx: Optional[int]
    x: float
    y: float
    smoothed_x: Optional[float]
    smoothed_y: Optional[float]
    score: float


# -----------------------------------------------------------------------------
# Helpers to ingest metadata
# -----------------------------------------------------------------------------


def load_segments(path: Path) -> List[Segment]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "segments" in data:
        raw_segments = data["segments"]
    elif isinstance(data, list):
        raw_segments = data
    else:
        raise ValueError("segments JSON must be a list or contain 'segments'")

    segments: List[Segment] = []
    for idx, entry in enumerate(raw_segments):
        start_index = int(entry.get("start_index", entry.get("start_frame", 0)))
        end_index = int(entry.get("end_index", entry.get("end_frame", start_index)))
        length = int(entry.get("length", end_index - start_index + 1))
        start_total = int(entry.get("start_total_frame", entry.get("start_frame", start_index)))
        end_total = int(entry.get("end_total_frame", entry.get("end_frame", end_index)))
        start_local = int(entry.get("start_frame", start_index))
        end_local = int(entry.get("end_frame", end_index))
        parquet_idx = entry.get("parquet_idx")
        parquet_idx = int(parquet_idx) if parquet_idx is not None else None
        task = entry.get("task")
        segments.append(
            Segment(
                segment_idx=int(entry.get("segment_idx", idx)),
                start_index=start_index,
                end_index=end_index,
                length=length,
                start_total_frame=start_total,
                end_total_frame=end_total,
                start_local_frame=start_local,
                end_local_frame=end_local,
                parquet_idx=parquet_idx,
                task=task,
            )
        )
    return segments


def expand_parquet_tokens(tokens: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for token in tokens:
        expanded = os.path.expanduser(token)
        candidate = Path(expanded)
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.parquet")))
            continue
        if candidate.is_file() and candidate.suffix == ".parquet":
            paths.append(candidate)
            continue
        for match in glob.glob(expanded, recursive=True):
            path = Path(match)
            if path.is_file() and path.suffix == ".parquet":
                paths.append(path)
    unique: List[Path] = []
    seen = set()
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _expand_glob(pattern: str) -> List[Path]:
    expanded = os.path.expanduser(pattern)
    matches = [Path(p) for p in glob.glob(expanded)]
    return sorted(path for path in matches if path.is_file())


def load_tracks(tracks_dir: Optional[str], combined_path: Optional[Path]) -> Dict[Tuple[int, int], TrackRecord]:
    if tracks_dir is None and combined_path is None:
        raise ValueError("Either --tracks-dir or --combined-tracks must be provided")

    records: Dict[Tuple[int, int], TrackRecord] = {}

    if combined_path is not None:
        entries = json.loads(combined_path.read_text())
        if not isinstance(entries, list):
            raise ValueError("combined tracks JSON must be a list")
        for entry in entries:
            seg_idx = int(entry["segment_idx"])
            seg_frame = int(entry["segment_frame_idx"])
            patch_idx = entry.get("patch_idx", 0)
            rec = TrackRecord(
                segment_idx=seg_idx,
                segment_frame_idx=seg_frame,
                global_frame_idx=int(entry["global_frame_idx"]),
                parquet_idx=entry.get("parquet_idx"),
                x=float(entry.get("x", 0.0)),
                y=float(entry.get("y", 0.0)),
                smoothed_x=_maybe_float(entry.get("smoothed_x")),
                smoothed_y=_maybe_float(entry.get("smoothed_y")),
                score=float(entry.get("score", 0.0)),
            )
            records[(seg_idx, seg_frame)] = rec
    if tracks_dir is not None:
        track_files: List[Path] = []
        if os.path.isdir(os.path.expanduser(tracks_dir)):
            track_files = sorted(Path(tracks_dir).glob("segment_*.json"))
        else:
            track_files = _expand_glob(tracks_dir)
        for file in track_files:
            seg_entries = json.loads(file.read_text())
            for entry in seg_entries:
                seg_idx = int(entry["segment_idx"])
                seg_frame = int(entry["segment_frame_idx"])
                rec = TrackRecord(
                    segment_idx=seg_idx,
                    segment_frame_idx=seg_frame,
                    global_frame_idx=int(entry["global_frame_idx"]),
                    parquet_idx=entry.get("parquet_idx"),
                    x=float(entry.get("x", 0.0)),
                    y=float(entry.get("y", 0.0)),
                    smoothed_x=_maybe_float(entry.get("smoothed_x")),
                    smoothed_y=_maybe_float(entry.get("smoothed_y")),
                    score=float(entry.get("score", 0.0)),
                )
                records[(seg_idx, seg_frame)] = rec
    return records


def _maybe_float(value):
    if value is None:
        return None
    return float(value)


# -----------------------------------------------------------------------------
# Annotation scanning and sampling
# -----------------------------------------------------------------------------


def compute_shard_offsets(parquet_files: List[Path]) -> List[Tuple[int, int, int]]:
    if pq is None:
        raise RuntimeError("pyarrow is required when working with parquet inputs")
    offsets: List[Tuple[int, int, int]] = []
    total = 0
    for idx, path in enumerate(parquet_files):
        pf = pq.ParquetFile(path)
        rows = pf.metadata.num_rows if pf.metadata is not None else 0
        offsets.append((idx, total, total + rows - 1))
        total += rows
    return offsets


def global_to_parquet(global_idx: int, shard_offsets: List[Tuple[int, int, int]]) -> Tuple[int, int]:
    for shard_idx, start, end in shard_offsets:
        if start <= global_idx <= end:
            return shard_idx, global_idx - start
    raise IndexError(f"Global frame {global_idx} not covered by provided parquet shards")


def find_segment_by_global(global_idx: int, segments: List[Segment]) -> Optional[Segment]:
    # segments expected sorted by start_total_frame
    lo, hi = 0, len(segments) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        seg = segments[mid]
        if seg.start_total_frame <= global_idx <= seg.end_total_frame:
            return seg
        if global_idx < seg.start_total_frame:
            hi = mid - 1
        else:
            lo = mid + 1
    return None


def uniform_sample(segment_start: int, segment_end: int, start_idx: int, max_samples: int) -> List[int]:
    total = segment_end - start_idx + 1
    if total <= 0:
        return []
    num = min(max_samples, total)
    if num == 1:
        return [start_idx]
    positions = np.linspace(start_idx, segment_end, num=num)
    indices = sorted({int(round(p)) for p in positions})
    # ensure exact boundaries
    if indices[0] != start_idx:
        indices[0] = start_idx
    if indices[-1] != segment_end:
        indices[-1] = segment_end
    return indices


def format_points(sample_list: List[Dict]) -> str:
    coords = []
    for sample in sample_list:
        x = sample.get("smoothed_x")
        y = sample.get("smoothed_y")
        if x is None or y is None:
            x = sample.get("x", 0.0)
            y = sample.get("y", 0.0)
        coords.append([int(round(x)), int(round(y))])
    return json.dumps(coords)


def normalize_conversation_value(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def update_conversation(value, points_str: str, strict: bool = False) -> str:
    """Replace the trajectory list attached to a known anchor sentence.

    We prefer replacing only the coordinate list that follows the sentence
    "The trajectory of the end effector in the first image is ..." to avoid
    touching other bracketed content (e.g., DEPTH tokens). If the anchor is
    not found, we fall back to replacing the first coordinate list anywhere; if
    that also fails, append the points at the end.
    """
    text = normalize_conversation_value(value)
    if text is None:
        return points_str

    # Regex for a coordinate list like [[x,y],[x,y],...] with flexible spaces
    coord = r"\[(?:\s*\[\s*\d+\s*,\s*\d+\s*\]\s*,?\s*)+\]"
    # Anchor phrase — case-insensitive, flexible whitespace
    # Allow optional punctuation (colon/dash) between 'is' and the list
    anchor = r"(The\s+trajectory\s+of\s+the\s+end\s+effector\s+in\s+the\s+first\s+image\s+is\s*[:\-–—]?\s*)"

    # 1) Try anchored replacement
    anchored_pat = re.compile(anchor + coord, flags=re.IGNORECASE | re.DOTALL)
    def _anchored_sub(m: re.Match) -> str:
        return m.group(1) + points_str

    new_text, n = anchored_pat.subn(_anchored_sub, text, count=1)
    if n > 0:
        return new_text

    # 2) Fallback: find anchor, then replace the next coord list after it
    anchor_only = re.search(anchor, text, flags=re.IGNORECASE)
    if anchor_only:
        start = anchor_only.end()
        coord_pat = re.compile(coord)
        coord_match = coord_pat.search(text, pos=start)
        if coord_match:
            return text[:coord_match.start()] + points_str + text[coord_match.end():]
        # If we're in strict mode and we found the anchor but not the coord list following it,
        # do not modify anything else.
        if strict:
            raise ValueError("Anchor found but no trajectory list followed to replace in conversations text")

    # 3) Last resort: replace the first coord list anywhere in the text
    first_coord_pat = re.compile(coord)
    coord_match = first_coord_pat.search(text)
    if coord_match:
        # Only allow this fallback when not strict, to avoid touching unrelated lists.
        if not strict:
            return text[:coord_match.start()] + points_str + text[coord_match.end():]
        raise ValueError("No anchored trajectory list found to replace in conversations text")

    # 4) Append or raise if nothing matched
    if strict:
        raise ValueError("No trajectory coordinate list found to replace in conversations text")
    return text.rstrip() + "\n" + points_str


def sample_annotations(
    segments: List[Segment],
    track_lookup: Dict[Tuple[int, int], TrackRecord],
    parquet_files: List[Path],
    annotation_column: str,
    max_samples: int,
    skip_duplicate_annotations: bool,
) -> Tuple[List[Dict], Dict[int, Dict[int, str]]]:
    shard_offsets = compute_shard_offsets(parquet_files)
    sorted_segments = sorted(segments, key=lambda s: s.start_total_frame)

    samples: List[Dict] = []
    updates: Dict[int, Dict[int, Dict[str, Optional[str]]]] = {}

    for shard_idx, path in enumerate(parquet_files):
        pf = pq.ParquetFile(path)
        metadata = pf.metadata
        num_groups = metadata.num_row_groups if metadata is not None else 1
        local_index = 0
        for rg_idx in range(num_groups):
            table = pf.read_row_group(rg_idx, columns=[annotation_column])
            rows = table.to_pylist()
            for row in rows:
                value = row.get(annotation_column) if isinstance(row, dict) else row
                global_idx = shard_offsets[shard_idx][1] + local_index
                segment = find_segment_by_global(global_idx, sorted_segments)
                if segment is None:
                    local_index += 1
                    continue
                segment_frame_idx = global_idx - segment.start_total_frame
                track_key = (segment.segment_idx, segment_frame_idx)
                track = track_lookup.get(track_key)
                if track is None:
                    local_index += 1
                    continue

                sample_indices = uniform_sample(
                    segment.start_total_frame,
                    segment.end_total_frame,
                    global_idx,
                    max_samples,
                )
                sample_list: List[Dict] = []
                for sample_global in sample_indices:
                    seg_frame_idx = sample_global - segment.start_total_frame
                    track_sample = track_lookup.get((segment.segment_idx, seg_frame_idx))
                    if track_sample is None:
                        continue
                    sample_parquet_idx, sample_local_idx = (segment.parquet_idx, None)
                    if segment.parquet_idx is not None:
                        sample_local_idx = segment.start_local_frame + seg_frame_idx
                    else:
                        sample_parquet_idx, sample_local_idx = global_to_parquet(sample_global, shard_offsets)
                    sample_list.append(
                        {
                            "segment_frame_idx": seg_frame_idx,
                            "global_frame_idx": sample_global,
                            "parquet_idx": sample_parquet_idx,
                            "local_row_idx": sample_local_idx,
                            "x": track_sample.x,
                            "y": track_sample.y,
                            "smoothed_x": track_sample.smoothed_x,
                            "smoothed_y": track_sample.smoothed_y,
                            "score": track_sample.score,
                        }
                    )

                points_str = format_points(sample_list)
                sample_entry = {
                    "parquet_idx": shard_idx,
                    "local_row_idx": local_index,
                    "parquet_path": str(path),
                    "global_frame_idx": global_idx,
                    "segment_idx": segment.segment_idx,
                    "segment_frame_idx": segment_frame_idx,
                    "annotation": value,
                    "task": segment.task,
                    "samples": sample_list,
                    "points_str": points_str,
                }
                samples.append(sample_entry)

                if value is None or value == "":
                    # Update conversations only
                    updates.setdefault(shard_idx, {})[local_index] = {
                        "annotation": None,
                        "conversation": points_str,
                    }
                else:
                    # Update annotation only
                    updates.setdefault(shard_idx, {})[local_index] = {
                        "annotation": points_str,
                    }
                local_index += 1
    return samples, updates


def write_updated_parquet(
    parquet_files: List[Path],
    updates: Dict[int, Dict[int, Dict[str, Optional[str]]]],
    annotation_column: str,
    conversation_column: Optional[str],
    output_dir: Optional[Path],
    overwrite: bool,
) -> None:
    if not updates:
        print("[INFO] No annotation updates to write")
        return
    if pq is None or pa is None:
        raise RuntimeError("pyarrow is required to update parquet files")

    for shard_idx, mapping in updates.items():
        path = parquet_files[shard_idx]
        print(f"[INFO] Updating parquet shard {path} with {len(mapping)} annotation rows")
        table = pq.read_table(path)
        if annotation_column not in table.column_names:
            raise ValueError(f"Column '{annotation_column}' missing from {path}")
        ann_col = table.column(annotation_column).to_pylist()
        conv_col = None
        conv_idx = None
        if conversation_column and conversation_column in table.column_names:
            conv_idx = table.column_names.index(conversation_column)
            conv_col = [normalize_conversation_value(v) for v in table.column(conversation_column).to_pylist()]

        for local_idx, payload in mapping.items():
            if local_idx < 0 or local_idx >= len(ann_col):
                raise IndexError(
                    f"Row {local_idx} out of range for parquet shard {path}"
                )
            annotation_val = payload.get("annotation")
            if annotation_val is not None:
                # Update both annotation and conversations only for rows with non-null annotations
                ann_col[local_idx] = annotation_val
                if conv_col is not None and payload.get("conversation") is not None:
                    conv_col[local_idx] = update_conversation(
                        conv_col[local_idx], payload["conversation"], strict=True
                    )

        new_table = table.set_column(
            table.column_names.index(annotation_column),
            annotation_column,
            pa.array(ann_col),
        )
        if conv_col is not None and conv_idx is not None:
            new_table = new_table.set_column(conv_idx, conversation_column, pa.array(conv_col))
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / path.name
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"Output parquet {out_path} already exists (use --overwrite-parquet)")
        else:
            if not overwrite and os.access(path, os.F_OK) and not os.access(path, os.W_OK):
                raise PermissionError(f"Cannot overwrite {path}; use --parquet-output-dir")
            out_path = path if overwrite else path.with_suffix(".annotated.parquet")
            if out_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Output parquet {out_path} already exists (use --overwrite-parquet or specify --parquet-output-dir)"
                )
        pq.write_table(new_table, out_path)
        print(f"[INFO] Wrote updated parquet shard to {out_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample tracking points per annotation")
    parser.add_argument("--segments-json", required=True)
    parser.add_argument("--tracks-dir", help="Directory containing per-segment track JSONs")
    parser.add_argument("--combined-tracks", help="Optional combined track JSON (overrides dir when clashes)")
    parser.add_argument("--parquet", action="append", required=True, help="Parquet file(s)/directories/globs")
    parser.add_argument("--annotation-column", default="annotation")
    parser.add_argument("--max-samples", type=int, default=5, help="Maximum samples per annotation")
    parser.add_argument(
        "--keep-duplicate-annotations",
        action="store_true",
        help="Disable duplicate annotation filtering (default skips duplicates)",
    )
    parser.add_argument("--conversation-column", default="conversations", help="Column for natural language annotations")
    parser.add_argument("--segments", nargs="*", type=int, help="Restrict to specific segment indices")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--write-updated-parquet",
        action="store_true",
        help="Replace annotation columns with sampled payloads",
    )
    parser.add_argument(
        "--parquet-output-dir",
        help="Directory for updated parquet shards (default: in-place .annotated.parquet)",
    )
    parser.add_argument(
        "--overwrite-parquet",
        action="store_true",
        help="Allow overwriting destination parquet files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if pq is None:
        raise RuntimeError("pyarrow is required for this script")

    segments = load_segments(Path(args.segments_json))
    if args.segments:
        segments = [seg for seg in segments if seg.segment_idx in set(args.segments)]
        if not segments:
            print("[WARN] No segments selected; exiting")
            return

    track_lookup = load_tracks(Path(args.tracks_dir) if args.tracks_dir else None, Path(args.combined_tracks) if args.combined_tracks else None)

    parquet_files = expand_parquet_tokens(args.parquet)
    if not parquet_files:
        raise FileNotFoundError("No parquet files found via --parquet")

    samples, updates = sample_annotations(
        segments=segments,
        track_lookup=track_lookup,
        parquet_files=parquet_files,
        annotation_column=args.annotation_column,
        max_samples=args.max_samples,
        skip_duplicate_annotations=not args.keep_duplicate_annotations,
    )

    Path(args.output).write_text(json.dumps(samples, indent=2))
    print(f"[INFO] Wrote {len(samples)} annotation samples to {args.output}")

    if args.write_updated_parquet:
        output_dir = Path(args.parquet_output_dir) if args.parquet_output_dir else None
        write_updated_parquet(
            parquet_files=parquet_files,
            updates=updates,
            annotation_column=args.annotation_column,
            conversation_column=args.conversation_column,
            output_dir=output_dir,
            overwrite=args.overwrite_parquet,
        )


if __name__ == "__main__":
    main()
