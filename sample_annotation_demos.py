#!/usr/bin/env python3
"""Utility to sample sharegpt-style demo data from annotation JSON + manifest CSV."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


ManifestEntry = Tuple[str, Optional[Path]]

MILESTONE_DESCRIPTIONS = {
    "PRE_GRASP": "the gripper needs to move towards the source object",
    "GRASP": "the gripper now should close",
    "PLACE": "the gripper needs to place the object",
    "POST_PLACE": "the object is placed and the gripper should be released",
}


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


def load_annotations(path: Path) -> Dict[str, Dict]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("Annotation file must be a JSON object mapping episode ids to entries.")
    return data


def load_manifest_lookup(path: Optional[Path]) -> Dict[Tuple[str, str], ManifestEntry]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"episode_id", "change_point", "png"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest CSV missing columns: {', '.join(sorted(missing))}")

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


def resolve_frame_image(
    *,
    episode_id: str,
    frame_id: str,
    image_root: Optional[Path],
    image_pattern: Optional[str],
    manifest_lookup: Dict[Tuple[str, str], ManifestEntry],
) -> Optional[str]:
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


def build_gpt_response(
    picked_entries: Sequence[Dict],
    resolved_paths: Sequence[Optional[str]],
) -> str:
    phrases: List[str] = []
    for entry, path in zip(picked_entries, resolved_paths):
        milestone = str(entry.get("milestone", "")).upper()
        description = MILESTONE_DESCRIPTIONS.get(milestone)
        reason = entry.get("reason") or ""

        if not description:
            description = f"the scene reaches milestone {milestone.lower()}"

        components = [description.rstrip(".")]
        if reason:
            components.append(reason.strip())
        if path:
            components.append(f"frame: {path}")

        phrases.append(
            ", ".join(components).rstrip(".")
        )

    if not phrases:
        return "I need more visual context to describe the manipulation steps."

    sentences: List[str] = []
    for idx, phrase in enumerate(phrases):
        prefix: str
        if idx == 0:
            prefix = "First"
        elif idx == len(phrases) - 1:
            prefix = "Finally"
        else:
            prefix = "Next"
        sentences.append(f"{prefix}, {phrase}.")

    return " ".join(sentences)


def build_sharegpt_entry(
    *,
    episode_id: str,
    task: str,
    picked_entries: Sequence[Dict],
    image_paths: Sequence[Optional[str]],
) -> Optional[Dict[str, object]]:
    if not picked_entries:
        return None

    init_entry = picked_entries[0]
    init_path = image_paths[0]
    if not init_path:
        return None

    human_prompt = (
        f"Given the task instruction '{task}' and current observation frame {init_path}, what are the next steps to finish?"
    )

    gpt_response = build_gpt_response(picked_entries[1:], image_paths[1:]) if len(picked_entries) > 1 else (
        "The scene only provides the initial observation without subsequent milestones."
    )

    images = [path for path in image_paths if path]
    if not images:
        return None

    return {
        "episode_id": episode_id,
        "human": human_prompt,
        "gpt": gpt_response,
        "images": images,
    }


def sample_demonstrations(
    annotations: Dict[str, Dict],
    manifest_lookup: Dict[Tuple[str, str], ManifestEntry],
    *,
    sample_size: int,
    seed: Optional[int],
    image_root: Optional[Path],
    image_pattern: Optional[str],
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    episode_ids = list(annotations.keys())
    rng.shuffle(episode_ids)

    results: List[Dict[str, object]] = []
    for ep_id in episode_ids:
        if len(results) >= sample_size:
            break

        entry = annotations.get(ep_id) or {}
        task = entry.get("task") or ""
        picked = entry.get("picked") or []
        if not isinstance(picked, list) or not picked:
            continue

        resolved_paths: List[Optional[str]] = []
        for item in picked:
            frame_id = str(item.get("frame_id", ""))
            resolved = resolve_frame_image(
                episode_id=str(ep_id),
                frame_id=frame_id,
                image_root=image_root,
                image_pattern=image_pattern,
                manifest_lookup=manifest_lookup,
            )
            resolved_paths.append(resolved)

        sharegpt_entry = build_sharegpt_entry(
            episode_id=str(ep_id),
            task=task,
            picked_entries=picked,
            image_paths=resolved_paths,
        )
        if sharegpt_entry:
            results.append(sharegpt_entry)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample ShareGPT-style conversations from annotations.")
    parser.add_argument("annotations", type=Path, help="Path to annotation JSON file.")
    parser.add_argument("--manifest_csv", type=Path, default=None, help="Optional manifest CSV mapping change points to image paths.")
    parser.add_argument("--image_root", type=Path, default=None, help="Optional base directory for relative image paths.")
    parser.add_argument("--image_pattern", type=str, default=None, help="Optional fallback pattern for images (keys: image_root, episode_id, frame_id).")
    parser.add_argument("--output", type=Path, default=Path("sharegpt_samples.json"), help="Destination JSON file for sampled data.")
    parser.add_argument("--samples", type=int, default=8, help="Number of episodes to sample.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    manifest_lookup = load_manifest_lookup(args.manifest_csv)
    image_root = args.image_root.resolve() if args.image_root else None

    samples = sample_demonstrations(
        annotations,
        manifest_lookup,
        sample_size=max(args.samples, 0),
        seed=args.seed,
        image_root=image_root,
        image_pattern=args.image_pattern,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(samples, fp, indent=2, ensure_ascii=False)

    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
