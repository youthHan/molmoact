"""Reconstruct LoRA adapters from a merged MolmoAct checkpoint.

Given a baseline checkpoint (pre-LoRA) and a merged checkpoint (baseline +
LoRA deltas fused in), this script factors the weight differences back into
LoRA matrices and writes them out as a PEFT adapter directory. You must
provide the LoRA hyperparameters that were used during tuning.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import torch

from olmo.train.checkpointer import load_model_state
from olmo.train.trainer_config import TrainConfig
from olmo.util import prepare_cli_environment, resource_path


try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "peft is required for reconstructing LoRA adapters. Install it via `pip install peft`."
    ) from exc


def _load_model(cfg: TrainConfig, checkpoint_dir: str) -> torch.nn.Module:
    with torch.device("meta"):
        model = cfg.model.build_model()

    model.to_empty(device=torch.device("cpu"))
    load_model_state(checkpoint_dir, model)
    return model


def _factor_linear_deltas(
    base_model: torch.nn.Module,
    merged_state: Dict[str, torch.Tensor],
    rank: int,
    alpha: float,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Return LoRA (A, B) tensors for each Linear weight delta."""

    adapters: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    base_params = dict(base_model.named_parameters())
    scale = alpha / rank

    for module_name, module in base_model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        weight_key = f"{module_name}.weight" if module_name else "weight"
        if weight_key not in base_params or weight_key not in merged_state:
            continue

        diff = (merged_state[weight_key] - base_params[weight_key]).float()
        if torch.count_nonzero(diff) == 0:
            continue

        # Truncated SVD gives us a stable low-rank factorisation.
        u, s, vT = torch.linalg.svd(diff, full_matrices=False)
        usable_rank = min(rank, diff.shape[0], diff.shape[1])
        if usable_rank == 0:
            continue

        u = u[:, :usable_rank]
        s = s[:usable_rank]
        v = vT[:usable_rank, :]

        # peft applies scale * (B @ A); undo the scale here.
        lora_B = (u * s) / scale
        lora_A = v

        adapters[module_name] = (
            lora_A.to(dtype=base_params[weight_key].dtype).contiguous(),
            lora_B.to(dtype=base_params[weight_key].dtype).contiguous(),
        )

    return adapters


def _populate_lora_layers(
    peft_model, adapters: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
) -> None:
    for name, module in peft_model.base_model.named_modules():
        if not hasattr(module, "lora_A") or name not in adapters:
            continue

        lora_A_tensor, lora_B_tensor = adapters[name]
        module.lora_A.default.weight.data.copy_(lora_A_tensor)
        module.lora_B.default.weight.data.copy_(lora_B_tensor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract LoRA adapter from merged MolmoAct weights")
    parser.add_argument("base_checkpoint", help="Checkpoint directory with pre-LoRA weights")
    parser.add_argument("merged_checkpoint", help="Merged checkpoint (base + LoRA deltas)")
    parser.add_argument("output_dir", help="Where to save the reconstructed LoRA adapter")
    parser.add_argument(
        "--config",
        help="Optional path to config.yaml; defaults to <base_checkpoint>/config.yaml",
    )
    parser.add_argument("--lora-rank", type=int, required=True, help="LoRA rank used during tuning")
    parser.add_argument("--lora-alpha", type=float, required=True, help="LoRA alpha used during tuning")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout hyperparameter")
    parser.add_argument(
        "--lora-bias",
        default="none",
        choices=["none", "all", "lora_only"],
        help="LoRA bias setting used during tuning",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_cli_environment()

    config_path = args.config or resource_path(args.base_checkpoint, "config.yaml")
    train_cfg: TrainConfig = TrainConfig.load(config_path)

    effective_alpha = min(args.lora_rank, args.lora_alpha)

    base_model = _load_model(train_cfg, args.base_checkpoint)
    merged_model = _load_model(train_cfg, args.merged_checkpoint)

    merged_state = dict(merged_model.named_parameters())
    del merged_model  # release memory

    adapters = _factor_linear_deltas(base_model, merged_state, args.lora_rank, effective_alpha)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=effective_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )

    peft_model = get_peft_model(base_model, lora_config)
    _populate_lora_layers(peft_model, adapters)

    os.makedirs(args.output_dir, exist_ok=True)
    peft_model.save_pretrained(args.output_dir)

    # Add a small helper file describing what we did.
    meta_path = os.path.join(args.output_dir, "extraction_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_checkpoint": os.path.abspath(args.base_checkpoint),
                "merged_checkpoint": os.path.abspath(args.merged_checkpoint),
                "lora_rank": args.lora_rank,
                "lora_alpha": effective_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_bias": args.lora_bias,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()

