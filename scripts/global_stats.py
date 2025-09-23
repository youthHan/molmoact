#!/usr/bin/env python3
"""
Merge several MolmoAct suites that are stored as Parquet shards.

For each suite:
  - read the Parquet files (only the text metadata; images stay lazily loaded),
  - decode the action token list to continuous values with that suite’s original
    norm_stats entry,
  - accumulate global q01/q99 statistics,
  - re-encode every action with the new global stats,
  - emit per-suite replacement maps you can apply on-the-fly when loading data.

Outputs:
  1. global_norm_stats.json            # plug this into model.yaml/config.json
  2. suite_token_rewrites.json         # map original token strings → new tokens
     (use it in your dataloader instead of rewriting the Parquet files)
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from datasets import Dataset, load_dataset
from transformers import Qwen2Tokenizer

# ------------------------------------------------------------
#  Config: point each suite at its Parquet shards + config file
# ------------------------------------------------------------

# All suites must share the same discretization bins that MolmoAct expects.
N_ACTION_BINS = 256

SUITES: Dict[str, Dict[str, object]] = {
    "libero_goal": {
        # glob or explicit list; HF datasets handles both
        "data_files": [
            "/mnt/bn/kinetics-lp-maliva/data/molmoact_data/allenai/libero/libero_goal/train-*.parquet",
        ],
        "config": "/mnt/bn/kinetics-lp-maliva-v6/pretrain_models/molmoact/allenai/MolmoAct-7B-D-LIBERO-Goal-0812/config.json",
        "norm_key": "libero_goal_no_noops_modified",
    },
    "libero_object": {
        "data_files": [
            "/mnt/bn/kinetics-lp-maliva/data/molmoact_data/allenai/libero/libero_object/train-*.parquet",
        ],
        "config": "/mnt/bn/kinetics-lp-maliva-v6/pretrain_models/molmoact/allenai/MolmoAct-7B-D-LIBERO-Object-0812/config.json",
        "norm_key": "libero_object_no_noops_modified",
    },
    "libero_spatial": {
        # glob or explicit list; HF datasets handles both
        "data_files": [
            "/mnt/bn/kinetics-lp-maliva/data/molmoact_data/allenai/libero/libero_spatial/train-*.parquet",
        ],
        "config": "/mnt/bn/kinetics-lp-maliva-v6/pretrain_models/molmoact/allenai/MolmoAct-7B-D-LIBERO-Spatial-0812/config.json",
        "norm_key": "libero_spatial_no_noops_modified",
    },
    "libero_10": {
        "data_files": [
            "/mnt/bn/kinetics-lp-maliva/data/molmoact_data/allenai/libero/libero_10/train-*.parquet",
        ],
        "config": "/mnt/bn/kinetics-lp-maliva-v6/pretrain_models/molmoact/allenai/MolmoAct-7B-D-LIBERO-Long-0812/config.json",
        "norm_key": "libero_10_no_noops_modified",
    },
    # add more suites here
}

# ------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------

ACTION_BLOCK_RE = re.compile(r"\[(?:[^][\n]|\"[^\"]*\"|'[^']*')+\]")

@dataclass
class Example:
    suite: str
    example_id: str | int
    original_tokens: Tuple[str, ...]
    continuous_action: np.ndarray


def load_norm_stats(config_path: Path, norm_key: str) -> Dict[str, np.ndarray]:
    cfg = json.loads(config_path.read_text())
    stats = cfg["norm_stats"][norm_key]["action"]
    mask = stats.get("mask", [True] * len(stats["q01"]))
    mask = np.asarray(mask, dtype=bool)
    mask = mask.copy()
    mask[-1] = False  # keep gripper (last dim) unnormalized, matching codebase
    return {
        "q01": np.asarray(stats["q01"], dtype=np.float32),
        "q99": np.asarray(stats["q99"], dtype=np.float32),
        "mask": mask,
    }


def extract_action_tokens_from_conversation(conversation: Dict) -> List[str]:
    """
    The Parquet column `conversations` is a dict with keys:
      - "from": ["human", "gpt", ...]
      - "value": [user_prompt, assistant_reply, ...]
    The action token list lives in the assistant reply (typically index 1).
    """
    replies: List[str] = conversation["value"]
    all_match_parts = []
    # Search the assistant responses from last to first in case of multi-turn
    for text in reversed(replies):
        for match in ACTION_BLOCK_RE.finditer(text):
            inner = match.group(0)[1:-1]
            parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
            if parts and not all(is_number(p) for p in parts):
                all_match_parts.append(parts)

    if len(all_match_parts) == 0:
        raise ValueError("No action token list found in conversation.")
    else:
        return all_match_parts


def is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def load_suite_rows(data_files: Iterable[str]) -> Dataset:
    """
    Load only the columns needed; images stay as lazy Arrow buffers so we avoid
    decoding them.
    """
    return load_dataset(
        "parquet",
        data_files=list(data_files),
        split="train",
        columns=["conversations"],  # drop heavy image columns
    )


def decode_action(
    tokens: Sequence[str],
    tokenizer: Qwen2Tokenizer,
    bin_centers: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> np.ndarray:
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = np.array(
        [tokenizer.vocab_size if idx is None else int(idx) for idx in ids],
        dtype=np.int64,
    )
    discretized = tokenizer.vocab_size - ids
    discretized = np.clip(discretized - 1, 0, bin_centers.shape[0] - 1)
    normalized = bin_centers[discretized]

    q01, q99, mask = stats["q01"], stats["q99"], stats["mask"]
    unnorm = 0.5 * (normalized + 1.0) * (q99 - q01) + q01
    return np.where(mask, unnorm, normalized).astype(np.float32)


def encode_action(
    action: np.ndarray,
    tokenizer: Qwen2Tokenizer,
    bin_centers: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> List[str]:
    q01, q99, mask = stats["q01"], stats["q99"], stats["mask"]

    normalized = action.copy()
    mask_idx = np.where(mask)[0]
    normalized[mask_idx] = (
        2.0
        * (np.clip(action[mask_idx], q01[mask_idx], q99[mask_idx]) - q01[mask_idx])
        / (q99[mask_idx] - q01[mask_idx])
        - 1.0
    )
    normalized = np.clip(normalized, -1.0, 1.0)

    # Choose the nearest bin center per dimension.
    distances = np.abs(normalized[:, None] - bin_centers[None, :])
    discretized = distances.argmin(axis=1)
    token_ids = tokenizer.vocab_size - (discretized + 1)
    return tokenizer.convert_ids_to_tokens(token_ids.tolist())

# ------------------------------------------------------------
#  Main
# ------------------------------------------------------------

def main() -> None:
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-7B")
    bins = np.linspace(-1.0, 1.0, N_ACTION_BINS)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    per_suite_examples: Dict[str, List[Example]] = defaultdict(list)
    all_actions: List[np.ndarray] = []

    # Step 1: decode every existing action with its suite stats.
    for suite, spec in SUITES.items():
        dataset = load_suite_rows(spec["data_files"])
        stats = load_norm_stats(Path(spec["config"]), spec["norm_key"])

        for idx, row in enumerate(dataset):
            all_tokens = extract_action_tokens_from_conversation(row["conversations"])
            for tokens in all_tokens:
                action = decode_action(tokens, tokenizer, bin_centers, stats)
                per_suite_examples[suite].append(
                    Example(
                        suite=suite,
                        example_id=idx,
                        original_tokens=tuple(tokens),
                        continuous_action=action,
                    )
                )
                all_actions.append(action)

    if not all_actions:
        raise RuntimeError("Nothing decoded; check data_files or parser.")

    # Step 2: compute global quantiles.
    actions_np = np.stack(all_actions, axis=0)

    q01 = np.quantile(actions_np, 0.01, axis=0)
    q99 = np.quantile(actions_np, 0.99, axis=0)
    stats_min = actions_np.min(axis=0)
    stats_max = actions_np.max(axis=0)
    stats_mean = actions_np.mean(axis=0)
    stats_std = actions_np.std(axis=0, ddof=0)

    global_mask = np.ones(actions_np.shape[1], dtype=bool)
    global_mask[-1] = False

    global_norm_stats = {
        "global": {
            "action": {
                "min": stats_min.tolist(),
                "max": stats_max.tolist(),
                "mean": stats_mean.tolist(),
                "std": stats_std.tolist(),
                "q01": q01.tolist(),
                "q99": q99.tolist(),
                # "mask": global_mask.tolist(),
            }
        }
    }

    # Step 3: re-encode with the new stats.
    encoding_stats = {"q01": q01, "q99": q99, "mask": global_mask}
    suite_token_rewrites: Dict[str, Dict[str, List[str]]] = defaultdict(dict)

    for suite, examples in per_suite_examples.items():
        for ex in examples:
            new_tokens = encode_action(
                ex.continuous_action, tokenizer, bin_centers, encoding_stats
            )
            # Store mapping from the original token tuple to the replacement.
            suite_token_rewrites[suite].setdefault(
                " ".join(ex.original_tokens), new_tokens
            )

    # Step 4: dump results.
    Path("global_norm_stats.json").write_text(
        json.dumps({"norm_stats": global_norm_stats}, indent=2)
    )
    Path("suite_token_rewrites.json").write_text(
        json.dumps(suite_token_rewrites, indent=2, ensure_ascii=False)
    )
    print("Wrote global_norm_stats.json and suite_token_rewrites.json.")

if __name__ == "__main__":
    main()
