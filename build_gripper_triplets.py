#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re, json, sys
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

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
    
    return direct_video_frame_with_ffmpeg(video_path, episode_id, rel_frame_idx)
    
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

def direct_video_frame_with_ffmpeg(video_path: str, episode_id: int, rel_frame_idx: int) -> Optional[Image.Image]:
    """使用FFmpeg从视频中提取帧"""
    # video_path = f"{root_dir}/videos/chunk-000/observation.images.image/episode_{episode_id:06d}.mp4"
    # if not os.path.exists(video_path):
    #     print(f"[DEBUG] 视频文件不存在: {video_path}")
    #     return None
    
    temp_dir = f"{root_dir}/temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    output_path = f"{temp_dir}/frame_{episode_id}_{rel_frame_idx}.png"
    
    if extract_frame_ffmpeg(video_path, rel_frame_idx, output_path):
        try:
            return Image.open(output_path)
        except Exception as e:
            print(f"[DEBUG] 无法打开提取的帧: {str(e)}")
            return None
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
    root_dir = args.root #"/mnt/bn/kinetics-lp-maliva/playground_projects/bagel/self-cook/libero_10_no_noops_1.0.0_lerobot/"
    
    # Try LeRobotDataset first (preferred), otherwise datasets.load_dataset
    lrd = try_load_lerobot_dataset(args.dataset, args.split, root_dir)#, args.local_files_only)
    if lrd is not None:
        # Make sure HF returns plain Python objects, not torch/PIL wrappers
        hf_ds = lrd.hf_dataset.with_format(None) if lrd is not None else hf_ds
        fps = getattr(lrd, "fps", None)
        print(f"[INFO] Loaded via LeRobotDataset. episodes={lrd.num_episodes} frames={lrd.num_frames} fps={fps}")
    else:
        print("[INFO] lerobot not found or failed to load; falling back to datasets.load_dataset.")
        hf_ds = load_dataset(args.dataset, split=args.split)

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
    if args.prefer_video_fallback and lrd is not None:
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

    if args.image_key is not None:
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

    # Group by episode
    ep_to_indices = group_indices_by_episode(hf_ds, ep_key, step_key)
    print(f"[INFO] Episodes detected: {len(ep_to_indices)}")

    episodes_winst = [json.loads(l) for l in open(f'{root_dir}/meta/episodes.jsonl','r').readlines()]
    episodes_winst_dict = {}
    inst_episodes_dict = {}
    for ep in episodes_winst:
        inst_episodes_dict.setdefault(ep['tasks'][0], []).append(ep['episode_index'])
        episodes_winst_dict[ep['episode_index']] = ep['tasks'][0]

    print("length: ", len(inst_episodes_dict))
    inst_done_mark = {inst: False for inst in inst_episodes_dict.keys()}  

    manifest_rows: List[Dict[str, Any]] = []
    triplet_records: List[Dict[str, Any]] = []

    episodes_processed = 0
    for ep_id, indices in tqdm(ep_to_indices.items(), desc="Episodes"):
        # print(ep_id)
        # if ep_id > 0:
        #     break
        if ep_id > 3000:
            break
        if args.max_episodes is not None and episodes_processed >= args.max_episodes:
            break

        inst = episodes_winst_dict[ep_id]
        if inst == "" or inst_done_mark[inst]:
            continue

        lines = [hf_ds[ind]['observation.state'][:3]+[hf_ds[ind]['observation.state'][6]-hf_ds[ind]['observation.state'][7]] for ind in indices]
        actions = [hf_ds[ind]['action'] for ind in indices]

        # cps = find_change_points(grip_bits)
        cps = [int(cp) for cp in detect_events_xyzg([l[0] for l in lines], [l[1] for l in lines], [l[2] for l in lines], [l[3] for l in lines],
                                    rel_height=0.05,
                                    merge_window=3.0,
                                    min_separation=3,
                                    fuse_window=5.0,
                                    visualize=True,              # << turn on visualization
                                    out_dir="./self_cook_demo_figs"             # << save PNGs to this folder (optional)
                                    )['fused'].t_peak.values.tolist()]
        # print(f"[DEBUG] episode {ep_id}: 检测到的变化点: {len(cps)}, 前10个值: {grip_bits[:10]}")
        
        if len(cps) < args.min_changes:
            print(f"[DEBUG] episode {ep_id}: 变化点数量不足 ({len(cps)} < {args.min_changes})")
            continue

        # Fetch first image + timestamp
        first_idx = indices[0]
        first_row = hf_ds[int(first_idx)]

        first_img = lerobot_video_frame(lrd, episode_id=ep_id, rel_frame_idx=0, video_key=camera_key)

        if first_img is None:
            # Cannot produce triplets without images
            print(f"[DEBUG] episode {ep_id}: 无法获取first_img，跳过这个episode")
            continue

        base = f"{dump_name}_ep{ep_id}_cp0"
        p_first = os.path.join(export_dir, f"{base}.jpg")
        first_img.save(p_first, format="JPEG", quality=95, optimize=True)
        ts_first = ts_get(hf_ds[int(first_idx)]) if ts_get else None

        manifest_rows.append({
            "dump_name": dump_name,
            "episode_id": ep_id,
            "change_point": 0,
            "ts": ts_first,
            "state": None,
            "action":  None,
            "png": p_first,
            "num_changes_in_episode": len(cps)+2,
        })

        # frame_map = {}
        video_path = resolve_video_path_from_lrd(lrd, ep_id, camera_key, root_dir)
        if args.prefer_video_fallback and video_path and os.path.exists(video_path):
            cache_dir = os.path.join(root_dir, "temp_frames")
            frame_map = extract_many_frames_ffmpeg(video_path, cps, cache_dir, ep_id)

        for cidx, cp in enumerate(cps):
            # prefer not to touch image column at all
            # img = Image.open(frame_map[cp]) if cp in frame_map else None

            # if img is None and args.image_key is not None:
            #     va = hf_ds[int(after_idx)][args.image_key]
            #     if isinstance(va, dict) and "path" in va:
            #         img_after = Image.open(va["path"])

            # include_tag = (grip_bits[rel_after] - grip_bits[rel_before]) > 0
            ts = ts_get(hf_ds[int(cp)]) if ts_get else None

            base = f"{dump_name}_ep{ep_id}_cp{cp}"
            p  = os.path.join(export_dir, f"{base}.jpg")

            # img.save(p, format="JPEG", quality=95, optimize=True)

            shutil.move(frame_map[cp], p)

            manifest_rows.append({
                "dump_name": dump_name,
                "episode_id": ep_id,
                "change_point": int(cp),
                "ts": ts,
                "state": lines[cp],
                "action":  actions[cp],
                "png": p,
                "num_changes_in_episode": len(cps)+2,
            })

        base_last = f"{dump_name}_ep{ep_id}_cp{len(indices)-1}"
        p_last = os.path.join(export_dir, f"{base_last}.png")
        img_last = lerobot_video_frame(lrd, episode_id=ep_id, rel_frame_idx=len(indices)-1, video_key=camera_key)
        img_last.save(p_last)
        manifest_rows.append({
            "dump_name": dump_name,
            "episode_id": ep_id,
            "change_point": len(indices)-1,
            "ts": None,
            "state": None,
            "action":  None,
            "png": p_last,
            "num_changes_in_episode": len(cps)+2,
        })

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
    #         "episodes_detected": len(ep_to_indices),
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
python build_gripper_triplets.py \
    --dataset /mnt/bn/kinetics-lp-maliva/playground_projects/bagel/self-cook/libero_10_no_noops_1.0.0_lerobot/   \
    --split "train"   \
    --out_dir outputs/libero_triplets_new_fix/ \
    --prefer_video_fallback   \
    --gripper_key "action.6"   \
    --min_changes 1   \
    --root  /mnt/bn/kinetics-lp-maliva/playground_projects/bagel/self-cook/libero_10_no_noops_1.0.0_lerobot/ \
    --dump_name libero_triplets_new_fix
"""