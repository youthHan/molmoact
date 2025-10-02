#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re, json, sys, glob
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence

import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pq = None

import torch
import shutil
# Core HF datasets
from datasets import load_dataset, Dataset, Features, Value, Image as HFImage
from xyzg_events import detect_events_xyzg

import os
os.environ.setdefault("DATASETS_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
# use memory-mapped Arrow (don’t copy into RAM on every read)
os.environ.setdefault("DATASETS_ARROW_DEFAULT_MEMORY_MAPPING", "1")
# cut chatter
os.environ.setdefault("DATASETS_VERBOSITY", "error")

root_dir = "/mnt/bn/kinetics-lp-maliva/playground_projects/bagel/self-cook/libero_10_no_noops_1.0.0_lerobot/"
# Optional LeRobot (preferred path if installed)
def try_load_lerobot_dataset(repo_or_path: str, split: str, root: Optional[str]):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        print(f"[DEBUG] 尝试加载LeRobot，repo_or_path={repo_or_path}, root={root}")
        lrd = LeRobotDataset(repo_or_path, root=root)
        print(f"[DEBUG] LeRobot加载成功: {type(lrd)}")
        return lrd
    except Exception as e:
        print(f"[DEBUG] LeRobot加载失败: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

# Optional OpenCV for video-frame fallback (only used if needed)
def try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None

# ------------------------ Utilities ------------------------

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def dotted_get(obj: Any, dotted: str, default=None) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur

def find_first_matching_key(keys: List[str], patterns: List[str]) -> Optional[str]:
    rx = [re.compile(p, re.IGNORECASE) for p in patterns]
    for k in keys:
        for r in rx:
            if r.search(k):
                return k
    return None

def find_first_matching_path(row: Dict[str, Any], patterns: List[str]) -> Optional[str]:
    flat = flatten_dict(row)
    return find_first_matching_key(list(flat.keys()), patterns)

def to_bool_from_numeric(x: Any, threshold: float = 0.5) -> Optional[int]:
    try:
        arr = np.asarray(x).astype(float)
    except Exception:
        return None
    if arr.shape == ():
        return int(arr > threshold)
    return int(arr.flat[0] > threshold)

def image_from_value(val: Any) -> Optional[Image.Image]:
    if val is None:
        return None
    if isinstance(val, Image.Image):
        return val
    if isinstance(val, (np.ndarray, list)):
        arr = np.array(val)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            return Image.fromarray(arr)
    return None

def resolve_image_key_and_camera(example: Dict[str, Any],
                                 override_key: Optional[str],
                                 override_cam: Optional[str]) -> Tuple[str, Optional[str]]:
    if override_key:
        return override_key, override_cam
    candidates = [
        "image", "images",
        "observation.image", "observation.images",
        "obs.image", "obs.images",
        "camera.image", "camera.images",
    ]
    for k in candidates:
        base = dotted_get(example, k, None) if "." in k else example.get(k)
        if base is not None:
            cam = None
            if isinstance(base, dict):
                cam = override_cam or (sorted(base.keys())[0] if base else None)
            return k, cam
    img_path = find_first_matching_path(example, [r"(^|\.)(images?|rgb)(_|\.)?", r"(^|\.)(cam|wrist|front).*(image)"])
    if img_path:
        base = dotted_get(example, img_path, None) if "." in img_path else example.get(img_path)
        cam = None
        if isinstance(base, dict):
            cam = override_cam or (sorted(base.keys())[0] if base else None)
        return img_path, cam
    raise KeyError("Could not auto-detect image key. Provide --image_key and optionally --image_camera.")

def get_image_from_row(row: Dict[str, Any], image_key: str, camera_key: Optional[str]) -> Optional[Image.Image]:
    val = dotted_get(row, image_key, None) if "." in image_key else row.get(image_key)
    if isinstance(val, dict):
        if camera_key is None:
            if not val:
                return None
            camera_key = sorted(val.keys())[0]
        val = val.get(camera_key)
    return image_from_value(val)

def detect_episode_key(example: Dict[str, Any], column_names: List[str]) -> Optional[str]:
    patterns = [
        r"(^|\.)(episode)(_)?(id|index)$", r"(^|\.)(traj(ectory)?_id)$",
        r"(^|\.)(sequence_id)$", r"(^|\.)(episode)$", r"(^|\.)(ep_id)$",
        r"(^|\.)(task_episode_id)$",
    ]
    k = find_first_matching_key(column_names, patterns)
    return k or find_first_matching_path(example, patterns)

def detect_step_key(example: Dict[str, Any], column_names: List[str]) -> Optional[str]:
    patterns = [
        r"(^|\.)(frame|step|t|time|index)(_)?(id|idx)?$",
        r"(^|\.)(step_index)$", r"(^|\.)(frame_index)$", r"(^|\.)(t_index)$"
    ]
    k = find_first_matching_key(column_names, patterns)
    return k or find_first_matching_path(example, patterns)

def detect_gripper_key(example: Dict[str, Any], column_names: List[str]) -> Optional[str]:
    patterns = [
        r"(^|\.)(action|act|observation|obs|state)\.(gripper.*(open|state|closed|width|command))",
        r"(^|\.)(gripper.*(open|state|closed|width|command))",
        r"(^|\.)(open|close)_gripper$",
    ]
    k = find_first_matching_key(column_names, patterns)
    return k or find_first_matching_path(example, patterns)

def detect_timestamp_key(example: Dict[str, Any], column_names: List[str]) -> Optional[str]:
    patterns = [
        r"(^|\.)(timestamp|time_stamp|time|ts)(_)?(ns|us|ms|s)?$",
        r"(^|\.)(observation|obs|meta)\.(timestamp|time)(_)?(ns|us|ms|s)?$",
        r"(^|\.)(frame|step)_time(_)?(ns|us|ms|s)?$",
    ]
    k = find_first_matching_key(column_names, patterns)
    return k or find_first_matching_path(example, patterns)

def resolve_gripper_getter(gripper_key: Optional[str], threshold: float):
    def getter(row: Dict[str, Any]) -> Optional[int]:
        if gripper_key:
            # 检查是否有数组索引格式 (例如 "action.6")
            if '.' in gripper_key and gripper_key.split('.')[-1].isdigit():
                base_key = '.'.join(gripper_key.split('.')[:-1])  # 例如 "action"
                index = int(gripper_key.split('.')[-1])  # 例如 6
                
                # 获取基础对象
                base_obj = dotted_get(row, base_key, None) if '.' in base_key else row.get(base_key)
                
                # 尝试索引访问
                if base_obj is not None and isinstance(base_obj, (list, tuple, np.ndarray, torch.Tensor)) and len(base_obj) > index:
                    #print(f"[DEBUG] 成功获取 {base_key}[{index}]={base_obj[index]}")
                    return to_bool_from_numeric(base_obj[index], threshold)
                else:
                    print(f"[DEBUG] 无法获取 {base_key}[{index}], base_obj={type(base_obj).__name__}, len={len(base_obj) if hasattr(base_obj, '__len__') else 'N/A'}")
            else:
                val = dotted_get(row, gripper_key, None) if "." in gripper_key else row.get(gripper_key)
                if val is None:
                    return None
                return to_bool_from_numeric(val, threshold)
                
        # fallback recursive scan
        flat = flatten_dict(row)
        for k, v in flat.items():
            if re.search(r"gripper", k, re.IGNORECASE):
                b = to_bool_from_numeric(v, threshold)
                if b is not None:
                    return b
        return None
    return getter
def resolve_timestamp_getter(timestamp_key: Optional[str]):
    def getter(row: Dict[str, Any]) -> Optional[float]:
        if timestamp_key:
            val = dotted_get(row, timestamp_key, None) if "." in timestamp_key else row.get(timestamp_key)
            if val is None:
                return None
            try:
                arr = np.asarray(val).astype(float)
                return float(arr.flat[0])
            except Exception:
                return None
        flat = flatten_dict(row)
        for k, v in flat.items():
            if re.search(r"(timestamp|time(_stamp)?|^time$|(^|\.)ts)(_|\.|$)", k, re.IGNORECASE):
                try:
                    arr = np.asarray(v).astype(float)
                    return float(arr.flat[0])
                except Exception:
                    continue
        return None
    return getter

def group_indices_by_episode(table, ep_key: Optional[str], step_key: Optional[str]) -> Dict[Any, List[int]]:
    if ep_key:
        groups: Dict[Any, List[int]] = {}
        for i in range(len(table)):
            row = table[i]
            ep_val = dotted_get(row, ep_key, None) if "." in ep_key else row.get(ep_key)
            ep_val = int(ep_val)
            if ep_val is None:
                flat = flatten_dict(row)
                ep_val = flat.get(ep_key, None)
            if ep_val is None:
                groups = {}
                break
            groups.setdefault(ep_val, []).append(i)
        if groups:
            return groups
    if step_key and (step_key in getattr(table, "column_names", [])):
        steps = table[step_key]
        groups, cur = {}, []
        ep_counter = 0
        for i, s in enumerate(steps):
            if (isinstance(s, (int, np.integer)) and s == 0) and cur:
                groups[ep_counter] = cur
                cur, ep_counter = [], ep_counter + 1
            cur.append(i)
        if cur:
            groups[ep_counter] = cur
        return groups
    return {0: list(range(len(table)))}

def find_change_points(bits: List[int]) -> List[int]:
    return [i for i in range(1, len(bits)) if bits[i] != bits[i - 1]]

def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def coerce_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
        return coerce_to_int(value[0])
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def extract_value(row: Dict[str, Any], key: Optional[str]):
    if key is None:
        return None
    if key and '.' in key:
        try:
            return dotted_get(row, key, None)
        except TypeError:
            return None
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except Exception:
        return None



def extract_state_line(row: Dict[str, Any]) -> Optional[List[float]]:
    state_val = extract_value(row, 'observation.state')
    if state_val is None:
        return None
    try:
        arr = np.asarray(state_val, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size < 8:
        return None
    try:
        return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[6] - arr[7])]
    except Exception:
        return None


def iterate_episode_batches(table, ep_key: Optional[str]):
    current_ep = None
    current_indices: List[int] = []
    current_rows: List[Dict[str, Any]] = []
    for idx in range(len(table)):
        row = table[idx]
        value = extract_value(row, ep_key) if ep_key else 0
        episode_id = coerce_to_int(value)
        if episode_id is None:
            episode_id = value if value is not None else current_ep
        if episode_id is None:
            episode_id = 0 if current_ep is None else current_ep
        if current_ep is None:
            current_ep = episode_id
        if episode_id != current_ep and current_rows:
            yield current_ep, current_indices, current_rows
            current_ep = episode_id
            current_indices = []
            current_rows = []
        current_indices.append(idx)
        current_rows.append(row)
    if current_rows:
        yield current_ep, current_indices, current_rows


# --------------- Parquet utilities ---------------


class ParquetShard:
    """Wrap a single parquet file and cache row groups for fast random access."""

    def __init__(self, path: Path, start_offset: int) -> None:
        if pq is None:
            raise RuntimeError("pyarrow is required to read parquet shards")
        self.path = Path(path)
        self.start = start_offset
        self._file = pq.ParquetFile(self.path)
        metadata = self._file.metadata
        self.rows = metadata.num_rows if metadata is not None else 0
        self.end = self.start + self.rows
        schema = self._file.schema_arrow
        self.columns = list(schema.names) if schema is not None else []

        self._row_groups: List[Tuple[int, int, int]] = []
        cumulative = 0
        if metadata is not None:
            for rg_idx in range(metadata.num_row_groups):
                rg_meta = metadata.row_group(rg_idx)
                length = rg_meta.num_rows
                self._row_groups.append((rg_idx, cumulative, length))
                cumulative += length
        else:
            self._row_groups.append((0, 0, self.rows))

        self._cache_rg: Optional[int] = None
        self._cache_table = None

    def _row_group_for(self, local_index: int) -> Tuple[int, int]:
        if local_index < 0 or local_index >= self.rows:
            raise IndexError(f"Row {local_index} out of bounds for shard {self.path}")
        for rg_idx, start, length in self._row_groups:
            if start <= local_index < start + length:
                return rg_idx, local_index - start
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


class LazyParquetDataset:
    """Expose a HuggingFace-like interface backed by raw parquet shards."""

    def __init__(self, files: Sequence[Path]):
        if pq is None:
            raise RuntimeError("pyarrow is required to read parquet shards")
        paths = [Path(f) for f in files]
        if not paths:
            raise ValueError("No parquet files provided")

        self.shards: List[ParquetShard] = []
        self.column_names: List[str] = []
        self.total_rows = 0

        seen: set[str] = set()
        offset = 0
        for path in paths:
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
                    self.column_names.append(col)

        if not self.shards or self.total_rows == 0:
            raise ValueError("The provided parquet files do not contain any rows")

        self._last_shard_idx = 0

    def __len__(self) -> int:
        return self.total_rows

    def _locate(self, index: int) -> Tuple[int, ParquetShard, int]:
        if index < 0 or index >= self.total_rows:
            raise IndexError(f"Index {index} out of bounds (len={self.total_rows})")
        shard_idx = self._last_shard_idx
        shard = self.shards[shard_idx]
        if shard.start <= index < shard.end:
            return shard_idx, shard, index - shard.start
        for shard_idx, shard in enumerate(self.shards):
            if shard.start <= index < shard.end:
                self._last_shard_idx = shard_idx
                return shard_idx, shard, index - shard.start
        raise IndexError(f"Unable to locate shard for index {index}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, slice):
            raise TypeError("Slicing is not supported for LazyParquetDataset")
        _, shard, local_index = self._locate(int(index))
        return shard.get_row(local_index)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


def _expand_parquet_tokens(tokens: Sequence[str]) -> List[Path]:
    matches: List[Path] = []
    for token in tokens:
        expanded = os.path.expanduser(token)
        if os.path.isdir(expanded):
            direct = sorted(Path(expanded).glob("*.parquet"))
            matches.extend(direct if direct else sorted(Path(expanded).rglob("*.parquet")))
            continue
        if os.path.isfile(expanded) and expanded.endswith(".parquet"):
            matches.append(Path(expanded))
            continue
        for candidate in glob.glob(expanded, recursive=True):
            if os.path.isfile(candidate) and candidate.endswith(".parquet"):
                matches.append(Path(candidate))
    unique = sorted({m.resolve() for m in matches})
    return unique


def resolve_parquet_inputs(dataset_arg: str, split: str, override: Optional[str]) -> List[Path]:
    if override:
        tokens = [tok.strip() for tok in override.split(",") if tok.strip()]
        files = _expand_parquet_tokens(tokens)
        if not files:
            raise FileNotFoundError(f"No parquet files located via --parquet_dir={override}")
        return files

    roots_to_try = []
    base = Path(os.path.expanduser(dataset_arg))
    roots_to_try.append(base)
    roots_to_try.append(base / split)
    roots_to_try.append(base / "data")
    roots_to_try.append(base / "data" / split)
    roots_to_try.append(base / split / "data")

    for root in roots_to_try:
        if not root.exists() or not root.is_dir():
            continue
        files = _expand_parquet_tokens([str(root / "*.parquet")])
        if not files:
            files = _expand_parquet_tokens([str(root / "**" / "*.parquet")])
        if files:
            return files
    raise FileNotFoundError(
        "Unable to locate parquet shards. Provide --parquet_dir with files or a directory containing .parquet files."
    )

# --------------- Video fallback for LeRobotDataset ---------------

def lerobot_video_frame(lrd, episode_id: int, rel_frame_idx: int, video_key: Optional[str] = None) -> Optional[Image.Image]:
    """
    Get a frame image from LeRobotDataset videos folder using OpenCV.
    Requires that videos are available locally (not just on hub).
    """
    # Find first frame idx of episode to compute absolute frame index if needed
    try:
        ep_from = int(lrd.episode_data_index["from"][episode_id])
        # ep_to = int(lrd.episode_data_index["to"][episode_id])  # not needed
    except Exception:
        return None

    # Prefer first available video key
    if video_key is None:
        try:
            vkeys = list(getattr(lrd.meta, "video_keys", []))
            if not vkeys:
                return None
            video_key = [key for key in vkeys if 'wrist' not in key][0]
        except Exception:
            return None

    # Translate (episode_id, video_key) to actual file path
    try:
        vpath = lrd.meta.get_video_file_path(episode_id, video_key)
        vpath = str(vpath)
    except Exception:
        return None

    video_path = os.path.join(root_dir, vpath)
    if not os.path.exists(video_path):
        print(f"[DEBUG] 视频文件不存在: {video_path}")
        return None
    
    return direct_video_frame_with_ffmpeg(video_path, episode_id, rel_frame_idx, cache_root=root_dir)
    
    # return None

    # # BGR->RGB
    # frame = frame[:, :, ::-1]
    # return Image.fromarray(frame)


def extract_frame_ffmpeg(video_path: str, frame_idx: int, output_path: str) -> bool:
    """使用FFmpeg提取视频帧"""
    try:
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", video_path,
            "-vf", f"select=eq(n\\,{frame_idx})",
            "-frames:v", "1",
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"[DEBUG] FFmpeg提取帧失败: {str(e)}")
        return False

def direct_video_frame_with_ffmpeg(video_path: str, episode_id: int, rel_frame_idx: int, cache_root: Optional[str] = None) -> Optional[Image.Image]:
    """使用FFmpeg从视频中提取帧"""
    base_dir = cache_root or root_dir or '.'
    temp_dir = os.path.join(base_dir, 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    output_path = os.path.join(temp_dir, f'frame_{episode_id}_{rel_frame_idx}.png')

    if extract_frame_ffmpeg(video_path, rel_frame_idx, output_path):
        try:
            img = Image.open(output_path).convert('RGB')
            img.load()
            return img
        except Exception as e:
            print(f"[DEBUG] 无法打开提取的帧: {str(e)}")
            return None
        finally:
            try:
                os.remove(output_path)
            except OSError:
                pass
    return None

def _ffmpeg_select_expr(frame_indices):
    parts = [f"eq(n\\,{int(i)})" for i in sorted(set(map(int, frame_indices)))]
    return "+".join(parts)

def extract_many_frames_ffmpeg(video_path: str, frame_indices, out_dir: str, ep_id: int):
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"ep{ep_id}_"
    out_pattern = os.path.join(out_dir, f"{prefix}%08d.png")
    cmd = [
        "ffmpeg","-y","-v","error","-i", video_path,
        "-vf", f"select='{_ffmpeg_select_expr(frame_indices)}',setpts=N/FRAME_RATE/TB",
        "-vsync","0", out_pattern
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    files = sorted(p for p in os.listdir(out_dir) if p.startswith(prefix) and p.endswith(".png"))
    mapping = {}
    for idx, fn in zip(sorted(set(map(int, frame_indices))), files):
        mapping[idx] = os.path.join(out_dir, fn)
    return mapping

def resolve_video_path_from_lrd(lrd, ep_id: int, camera_key: str | None, root_dir: str):
    if lrd is None:
        return None
    try:
        vkey = camera_key
        if vkey is None:
            vkeys = list(getattr(lrd.meta, "video_keys", []))
            vkey = [k for k in vkeys if 'wrist' not in k][0] if vkeys else None
        if vkey is None:
            return None
        vpath = str(lrd.meta.get_video_file_path(ep_id, vkey))
        return os.path.join(root_dir, vpath)
    except Exception:
        return None


def resolve_episode_video_path(ep_id: int, camera_key: Optional[str], lrd, root_dir: Optional[str],
                                video_dir: Optional[str], video_pattern: Optional[str]) -> Optional[str]:
    path = resolve_video_path_from_lrd(lrd, ep_id, camera_key, root_dir) if root_dir else None
    if path:
        return path

    search_root = video_dir or (os.path.join(root_dir, 'videos') if root_dir else None)
    if search_root is None:
        return None
    root_path = Path(os.path.abspath(search_root))
    if not root_path.exists():
        return None

    pattern = video_pattern or '**/episode_{episode:06d}.mp4'
    camera_variants = []
    if camera_key:
        camera_variants.extend([camera_key, sanitize_name(camera_key), camera_key.split('.')[-1]])
    camera_variants.append('')

    seen = set()
    for camera_variant in camera_variants:
        ctx = {
            'episode': ep_id,
            'camera': camera_variant,
            'camera_slug': sanitize_name(camera_variant) if camera_variant else '',
            'camera_basename': camera_variant.split('.')[-1] if camera_variant else '',
        }
        try:
            formatted = pattern.format(**ctx)
        except KeyError:
            formatted = pattern.format(episode=ep_id)
        if formatted in seen:
            continue
        seen.add(formatted)
        formatted_path = Path(formatted)
        if formatted_path.is_absolute():
            if formatted_path.exists():
                return str(formatted_path)
            continue
        matches = sorted(root_path.glob(formatted))
        if matches:
            return str(matches[0])
    return None
# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Build image triplets around gripper flips (LeRobot 2.x+ compatible).")
    ap.add_argument("--dataset", type=str, required=True, help="HF repo id or local dir (e.g., IPEC-COMMUNITY/xxxx)")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out_dir", type=str, required=True, help="Output root directory")
    ap.add_argument("--dump_name", type=str, default=None, help="Custom name for subfolder/prefix (default: derived)")
    ap.add_argument("--image_key", type=str, default=None, help="Dotted key to image or dict of cameras (if present)")
    ap.add_argument("--image_camera", type=str, default=None, help="Camera name if image key is a dict")
    ap.add_argument("--gripper_key", type=str, default=None, help="Dotted key for gripper signal")
    ap.add_argument("--timestamp_key", type=str, default=None, help="Dotted key for per-frame timestamp")
    ap.add_argument("--gripper_threshold", type=float, default=0.5, help="Binarize numeric gripper")
    ap.add_argument("--min_changes", type=int, default=0, help="Keep episodes with at least this many flips")
    ap.add_argument("--max_episodes", type=int, default=None, help="Cap processed episodes (debug)")
    ap.add_argument("--prefer_video_fallback", action="store_true",
                    help="If images not in table but videos exist, decode from videos (needs opencv).")
    ap.add_argument("--use_parquet_reader", action="store_true",
                    help="Read directly from raw parquet shards instead of HuggingFace datasets (requires pyarrow).")
    ap.add_argument("--parquet_dir", type=str, default=None,
                    help="Optional comma-separated list of parquet files or directories to scan when using --use_parquet_reader.")
    ap.add_argument("--video_dir", type=str, default=None,
                    help="Root directory that stores episode videos when not using LeRobotDataset.")
    ap.add_argument("--video_pattern", type=str, default="**/episode_{episode:06d}.mp4",
                    help="Glob pattern (relative to --video_dir) to locate videos; supports {episode}, {camera}, {camera_slug}, {camera_basename}.")
    # LeRobotDataset loader options
    ap.add_argument("--root", type=str, default=None, help="Local root for LeRobotDataset")
    ap.add_argument("--local_files_only", action="store_true", help="Do not fetch from hub in LeRobotDataset loader")
    ap.add_argument("--video_decoder", type=str, choices=["opencv", "ffmpeg", "auto"], default="auto",
                help="指定视频解码器: opencv, ffmpeg, 或auto(先尝试opencv，失败后使用ffmpeg)")
    args = ap.parse_args()

    default_dump = sanitize_name(args.dataset.split("/")[-1]) if "/" in args.dataset else sanitize_name(args.dataset)
    dump_name = sanitize_name(args.dump_name) if args.dump_name else default_dump
    out_root = os.path.join(args.out_dir, dump_name)
    export_dir = os.path.join(out_root, "images")
    os.makedirs(export_dir, exist_ok=True)

    global root_dir
    root_dir = args.root or (args.dataset if os.path.isdir(args.dataset) else None)
    if root_dir is not None:
        root_dir = os.path.abspath(root_dir)
    elif args.video_dir:
        root_dir = os.path.abspath(args.video_dir)

    if args.use_parquet_reader:
        if pq is None:
            print('[ERROR] pyarrow is required when --use_parquet_reader is set.')
            sys.exit(1)
        parquet_files = resolve_parquet_inputs(args.dataset, args.split, args.parquet_dir)
        print(f'[INFO] Reading {len(parquet_files)} parquet shards directly from disk.')
        hf_ds = LazyParquetDataset(parquet_files)
        lrd = None
    else:
        lrd = try_load_lerobot_dataset(args.dataset, args.split, root_dir)
        if lrd is not None:
            hf_ds = lrd.hf_dataset.with_format(None)
            fps = getattr(lrd, 'fps', None)
            print(f"[INFO] Loaded via LeRobotDataset. episodes={lrd.num_episodes} frames={lrd.num_frames} fps={fps}")
        else:
            print('[INFO] lerobot not found or failed to load; falling back to datasets.load_dataset.')
            hf_ds = load_dataset(args.dataset, split=args.split)
            try:
                hf_ds = hf_ds.with_format('python')
            except Exception:
                try:
                    hf_ds = hf_ds.with_format(None)
                except Exception:
                    pass

    # If LeRobot has any transforms, disable them to avoid Python-side work
    try:
        if hasattr(lrd, "transform"):
            lrd.transform = None
    except Exception:
        pass

    if len(hf_ds) == 0:
        print("[ERROR] Split is empty.")
        sys.exit(1)

    # If you plan to use videos anyway, completely avoid touching the image column:
    if args.prefer_video_fallback:
        # Force video-only path; skip LeRobot/HF image reads altogether
        args.image_key = None

    column_names = list(hf_ds.column_names)
    probe_row = hf_ds[0]

    # Detect keys (robust across variants)
    ep_key = detect_episode_key(probe_row, column_names)
    step_key = detect_step_key(probe_row, column_names)
    image_key, camera_key = None, None
    try:
        image_key, camera_key = resolve_image_key_and_camera(probe_row, args.image_key, args.image_camera)
    except Exception:
        # It may be a pure-video dataset; image extraction will rely on video fallback if requested.
        if not args.prefer_video_fallback or lrd is None:
            # If we can't fallback to videos, we must have an image key.
            if args.image_key is None:
                print("[WARN] Could not detect image key. If your dataset does not store per-frame images,"
                      " pass --prefer_video_fallback (requires local videos) or supply --image_key explicitly.")
            image_key = args.image_key  # could still be None

    if args.image_key is not None and hasattr(hf_ds, 'cast_column'):
        try:
            # Return dicts/paths, not PIL (decoding happens only when you save)
            hf_ds = hf_ds.cast_column(args.image_key, HFImage(decode=False))
        except Exception:
            pass

    grip_key = args.gripper_key or detect_gripper_key(probe_row, column_names)
    ts_key = args.timestamp_key or detect_timestamp_key(probe_row, column_names)

    grip_get = resolve_gripper_getter(grip_key, args.gripper_threshold)
    ts_get = resolve_timestamp_getter(ts_key)

    print(f"[INFO] episode_key={ep_key}  step_key={step_key}")
    print(f"[INFO] image_key={image_key}  camera={camera_key}")
    print(f"[INFO] gripper_key={grip_key}  threshold={args.gripper_threshold}")
    print(f"[INFO] timestamp_key={ts_key}")

    # Group by episode / instruction metadata
    episodes_winst_dict: Dict[Any, str] = {}
    inst_episodes_dict: Dict[str, List[Any]] = {}
    inst_done_mark: Dict[str, bool] = {}

    episodes_meta_path = None
    if root_dir:
        episodes_meta_path = Path(root_dir) / 'meta' / 'episodes.jsonl'

    if episodes_meta_path and episodes_meta_path.exists():
        with episodes_meta_path.open('r') as meta_f:
            for line in meta_f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep_idx_val = coerce_to_int(entry.get('episode_index'))
                if ep_idx_val is None:
                    continue
                tasks = entry.get('tasks') or []
                instruction = tasks[0] if tasks else ''
                inst_episodes_dict.setdefault(instruction, []).append(ep_idx_val)
                episodes_winst_dict[ep_idx_val] = instruction
        inst_done_mark = {inst: False for inst in inst_episodes_dict if inst}
        print(f"[INFO] Loaded instruction metadata for {len(inst_done_mark)} tasks from episodes.jsonl")
    else:
        if root_dir:
            print(f"[WARN] episodes.jsonl not found under {root_dir}; proceeding without instruction filtering.")
        else:
            print('[WARN] No dataset root provided; instruction metadata will be skipped.')

    manifest_rows: List[Dict[str, Any]] = []
    triplet_records: List[Dict[str, Any]] = []

    episode_iter = iterate_episode_batches(hf_ds, ep_key)
    estimated_total = len(episodes_winst_dict) if episodes_winst_dict else None
    episodes_processed = 0
    episode_progress = tqdm(episode_iter, desc="Episodes", total=estimated_total)
    for episode_idx, (raw_episode_id, indices, rows) in enumerate(episode_progress):
        if not indices or not rows:
            continue
        if args.max_episodes is not None and episodes_processed >= args.max_episodes:
            break

        ep_candidate = coerce_to_int(raw_episode_id)
        ep_id = ep_candidate if ep_candidate is not None else (raw_episode_id if raw_episode_id is not None else episode_idx)
        if isinstance(ep_id, np.integer):
            ep_id = int(ep_id)

        inst = episodes_winst_dict.get(ep_id)
        if inst_done_mark:
            if not inst or inst_done_mark.get(inst):
                continue

        state_lines: List[List[float]] = []
        actions: List[Any] = []
        timestamps: List[Optional[float]] = []
        missing_state = False
        for row in rows:
            state_line = extract_state_line(row)
            if state_line is None:
                missing_state = True
                break
            state_lines.append(state_line)
            actions.append(extract_value(row, 'action'))
            timestamps.append(ts_get(row) if ts_get else None)

        if missing_state or not state_lines:
            print(f"[DEBUG] episode {ep_id}: missing state data, skipping")
            continue

        try:
            event_info = detect_events_xyzg(
                [line[0] for line in state_lines],
                [line[1] for line in state_lines],
                [line[2] for line in state_lines],
                [line[3] for line in state_lines],
                rel_height=0.05,
                merge_window=3.0,
                min_separation=3,
                fuse_window=5.0,
                visualize=True,
                out_dir="./self_cook_demo_figs"
            )
            cps = [int(cp) for cp in event_info['fused'].t_peak.values.tolist()]
        except Exception as exc:
            print(f"[WARN] episode {ep_id}: change-point detection failed ({exc})")
            continue

        if len(cps) < args.min_changes:
            print(f"[DEBUG] episode {ep_id}: 变化点数量不足 ({len(cps)} < {args.min_changes})")
            continue

        ep_video_idx = coerce_to_int(ep_id)
        video_path = None
        if isinstance(ep_video_idx, int):
            video_path = resolve_episode_video_path(ep_video_idx, camera_key, lrd, root_dir, args.video_dir, args.video_pattern)
        if args.prefer_video_fallback and isinstance(ep_video_idx, int) and video_path is None and image_key is None:
            print(f"[WARN] episode {ep_id}: no video found; dataset images unavailable")
        frame_map: Dict[int, str] = {}
        cache_root = root_dir if root_dir else out_root
        if args.prefer_video_fallback and video_path and isinstance(ep_video_idx, int):
            cache_dir = os.path.join(cache_root, 'temp_frames')
            try:
                frame_map = extract_many_frames_ffmpeg(video_path, cps, cache_dir, ep_video_idx)
            except Exception as exc:
                frame_map = {}
                print(f"[WARN] episode {ep_id}: failed to batch-extract frames ({exc})")

        first_img = None
        if isinstance(ep_video_idx, int):
            first_img = lerobot_video_frame(lrd, episode_id=ep_video_idx, rel_frame_idx=0, video_key=camera_key)
        if first_img is None and video_path and isinstance(ep_video_idx, int):
            first_img = direct_video_frame_with_ffmpeg(video_path, ep_video_idx, 0, cache_root=cache_root)
        if first_img is None and image_key is not None:
            first_img = get_image_from_row(rows[0], image_key, camera_key)
        if first_img is None:
            print(f"[DEBUG] episode {ep_id}: 无法获取first_img，跳过这个episode")
            continue

        base_first = f"{dump_name}_ep{ep_id}_cp0"
        p_first = os.path.join(export_dir, f"{base_first}.jpg")
        first_img.save(p_first, format='JPEG', quality=95, optimize=True)
        ts_first = timestamps[0] if timestamps else None

        manifest_rows.append({
            'dump_name': dump_name,
            'episode_id': ep_id,
            'change_point': 0,
            'ts': ts_first,
            'state': None,
            'action': None,
            'png': p_first,
            'num_changes_in_episode': len(cps) + 2,
        })

        for cp in cps:
            if cp >= len(state_lines):
                continue
            ts_val = timestamps[cp] if cp < len(timestamps) else None
            base_cp = f"{dump_name}_ep{ep_id}_cp{cp}"
            p_cp = os.path.join(export_dir, f"{base_cp}.jpg")

            if frame_map and cp in frame_map:
                shutil.move(frame_map[cp], p_cp)
            else:
                img_cp = None
                if isinstance(ep_video_idx, int):
                    img_cp = lerobot_video_frame(lrd, episode_id=ep_video_idx, rel_frame_idx=cp, video_key=camera_key)
                if img_cp is None and video_path and isinstance(ep_video_idx, int):
                    img_cp = direct_video_frame_with_ffmpeg(video_path, ep_video_idx, cp, cache_root=cache_root)
                if img_cp is None and image_key is not None:
                    img_cp = get_image_from_row(rows[cp], image_key, camera_key)
                if img_cp is None:
                    continue
                img_cp.save(p_cp, format='JPEG', quality=95, optimize=True)

            manifest_rows.append({
                'dump_name': dump_name,
                'episode_id': ep_id,
                'change_point': int(cp),
                'ts': ts_val,
                'state': state_lines[cp],
                'action': actions[cp] if cp < len(actions) else None,
                'png': p_cp,
                'num_changes_in_episode': len(cps) + 2,
            })

        last_idx = len(rows) - 1
        base_last = f"{dump_name}_ep{ep_id}_cp{last_idx}"
        p_last = os.path.join(export_dir, f"{base_last}.png")
        last_img = None
        if isinstance(ep_video_idx, int):
            last_img = lerobot_video_frame(lrd, episode_id=ep_video_idx, rel_frame_idx=last_idx, video_key=camera_key)
        if last_img is None and video_path and isinstance(ep_video_idx, int):
            last_img = direct_video_frame_with_ffmpeg(video_path, ep_video_idx, last_idx, cache_root=cache_root)
        if last_img is None and frame_map and last_idx in frame_map:
            last_img = Image.open(frame_map[last_idx])
        if last_img is not None:
            last_img.save(p_last)
            manifest_rows.append({
                'dump_name': dump_name,
                'episode_id': ep_id,
                'change_point': last_idx,
                'ts': None,
                'state': None,
                'action': None,
                'png': p_last,
                'num_changes_in_episode': len(cps) + 2,
            })

        if inst_done_mark and inst:
            inst_done_mark[inst] = True

        episodes_processed += 1
    # Save CSV
    os.makedirs(out_root, exist_ok=True)
    manifest_path = os.path.join(out_root, f"{dump_name}_triplets_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"[OK] CSV manifest written to: {manifest_path}")

    # # Save HF dataset
    # feats = Features({
    #     "dump_name": Value("string"),
    #     "episode_id": Value("string"),
    #     "change_point": Value("int32"),
    #     "idx_first": Value("int64"),
    #     "idx_before": Value("int64"),
    #     "idx_after": Value("int64"),
    #     "ts_first": Value("float64"),
    #     "ts_before": Value("float64"),
    #     "ts_after": Value("float64"),
    #     "gripper_before": Value("int8"),
    #     "gripper_after": Value("int8"),
    #     "num_changes_in_episode": Value("int32"),
    #     "image_first": HFImage(),
    #     "image_before": HFImage(),
    #     "image_after": HFImage(),
    # })
    # out_ds = Dataset.from_list(triplet_records, features=feats)
    # ds_path = os.path.join(out_root, f"{dump_name}_triplets_hf_dataset")
    # out_ds.save_to_disk(ds_path)
    # print(f"[OK] HF Dataset saved to: {ds_path}")

    # # Save summary JSON
    # with open(os.path.join(out_root, f"{dump_name}_summary.json"), "w") as f:
    #     json.dump({
    #         "dataset": args.dataset,
    #         "split": args.split,
    #         "dump_name": dump_name,
    #         "episodes_detected": episodes_processed,
    #         "triplets": len(triplet_records),
    #         "min_changes": args.min_changes,
    #         "image_key": image_key,
    #         "camera": camera_key,
    #         "gripper_key": grip_key,
    #         "timestamp_key": ts_key,
    #         "gripper_threshold": args.gripper_threshold,
    #         "used_lerobot": (lrd is not None),
    #         "video_fallback": bool(args.prefer_video_fallback),
    #         "out_root": out_root,
    #     }, f, indent=2)
    # print("[DONE] Triplet extraction complete.")

if __name__ == "__main__":
    main()

"""
# Example: stream directly from parquet shards (no HF dataset preload)
python build_gripper_triplets.py \
    --dataset /mnt/bn/kinetics-lp-maliva/playground_projects/bagel/self-cook/libero_10_no_noops_1.0.0_lerobot/ \
    --split train \
    --out_dir outputs/libero_triplets_streaming/ \
    --dump_name libero_triplets_streaming \
    --use_parquet_reader \
    --parquet_dir /mnt/bn/kinetics-lp-maliva/playground_projects/bagel/self-cook/libero_10_no_noops_1.0.0_lerobot/data/train \
    --prefer_video_fallback \
    --video_dir /mnt/bn/kinetics-lp-maliva/playground_projects/bagel/self-cook/libero_10_no_noops_1.0.0_lerobot/videos \
    --video_pattern "chunk-*/observation.images.image/episode_{episode:06d}.mp4" \
    --gripper_key action.6 \
    --min_changes 1

# Example: rely on LeRobot's HF dataset (images already materialised)
python build_gripper_triplets.py \
    --dataset IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
    --split train \
    --out_dir outputs/libero_triplets_hf/ \
    --dump_name libero_triplets_hf \
    --root /mnt/bn/kinetics-lp-maliva/playground_projects/bagel/self-cook/libero_10_no_noops_1.0.0_lerobot/ \
    --image_key observation.images.image \
    --gripper_key action.6 \
    --min_changes 1
"""
