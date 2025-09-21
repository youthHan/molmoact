from pathlib import Path
import torch
from olmo.train.distributed_checkpointing import unshard_checkpoint
from olmo.train.checkpointer import Checkpointer, load_model_state_unsharded, load_model_state_hf, is_unsharded_checkpoint, is_hf_checkoint
from olmo.train.trainer_config import TrainConfig

# src = Path("checkpoints/libero-all-b100-10k-bs256x4-wconnector-repl/step6000/model_and_optim")          # sharded checkpoint
# dst = src.with_name(src.name + "-unsharded")          # new output dir
# unshard_checkpoint(src, dst, optim=False)             # set True if you also want optim.pt

from peft import LoraConfig, get_peft_model, PeftModel

cfg: TrainConfig = TrainConfig.load("checkpoints/libero-all-b100-10k-bs256x4-wconnector-repl/step6000/config.yaml")
model_cfg = cfg.model
with torch.device("meta"):
    olmo_model = model_cfg.build_model()                    # create Molmo
olmo_model.to_empty(device=torch.device("cpu"))

lora_cfg = LoraConfig(
                    r=model_cfg.lora_rank,
                    lora_alpha=min(model_cfg.lora_rank, model_cfg.lora_alpha),
                    # target_modules=find_all_linear_names(fsdp_model, model_cfg),
                    target_modules="all-linear",
                    lora_dropout=model_cfg.lora_dropout,
                    bias=model_cfg.lora_bias,
                    init_lora_weights="gaussian",
                )                  # same r/alpha/targets used in training
model = get_peft_model(olmo_model, lora_cfg)
load_model_state_unsharded("checkpoints/libero-all-b100-10k-bs256x4-wconnector-repl/step6000-unsharded", model) # or load sharded checkpoint
model = PeftModel.from_pretrained(model, "checkpoints/libero-all-b100-10k-bs256x4-wconnector-repl/step6000-lora")
