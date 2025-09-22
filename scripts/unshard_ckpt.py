from pathlib import Path
import torch
from olmo.train.distributed_checkpointing import unshard_checkpoint
from olmo.train.checkpointer import Checkpointer, load_model_state_unsharded, load_model_state_hf, is_unsharded_checkpoint, is_hf_checkoint, save_unsharded
from olmo.train.trainer_config import TrainConfig

base_dir = "checkpoints/libero-goal-b100-40k-bs32x4-repl/step30000"
src = Path(f"{base_dir}/model_and_optim")          # sharded checkpoint
dst = Path(f"{base_dir}-unsharded")          # new output dir
unshard_checkpoint(src, dst, optim=False)             # set True if you also want optim.pt

from peft import LoraConfig, get_peft_model, PeftModel

cfg: TrainConfig = TrainConfig.load(f"{base_dir}/config.yaml")
model_cfg = cfg.model
with torch.device("meta"):
    base_model = model_cfg.build_model()                    # create Molmo
base_model.to_empty(device=torch.device("cpu"))

lora_cfg = LoraConfig(
                    r=model_cfg.lora_rank,
                    lora_alpha=min(model_cfg.lora_rank, model_cfg.lora_alpha),
                    # target_modules=find_all_linear_names(fsdp_model, model_cfg),
                    target_modules="all-linear",
                    lora_dropout=model_cfg.lora_dropout,
                    bias=model_cfg.lora_bias,
                    init_lora_weights="gaussian",
                )                  # same r/alpha/targets used in training
peft_model = get_peft_model(base_model, lora_cfg)

# load the PEFT-structured checkpoint
load_model_state_unsharded(f"{base_dir}-unsharded", peft_model)

# fuse LoRA into the base and drop the wrapper
merged = peft_model.merge_and_unload()  # returns the plain Molmo module
out_dir = f"{base_dir}-merged"
save_unsharded(
    out_dir,
    merged,
    optim=None,                  # pass your optimizer if you also want its state
    config=cfg,                  # the TrainConfig (so config.yaml is written)
    overwrite=True,
)
"""
python3 -m olmo.hf_model.molmoact.convert_molmoact_to_hf  \
    /mnt/bn/kinetics-lp-maliva/playground_projects/MolmoAct/checkpoints/libero-all-b100-10k-bs256x4-wconnector-repl/step10000-merged \
    /mnt/bn/kinetics-lp-maliva/playground_projects/MolmoAct/checkpoints/libero-all-b100-10k-bs256x4-wconnector-repl/step10000-merged-hf \
    demo_role \
    --precision bf16

python3 -m olmo.hf_model.molmoact.convert_molmoact_to_hf \
    /mnt/bn/kinetics-lp-maliva/playground_projects/MolmoAct/checkpoints/libero-all-b100-10k-bs256x4-wconnector-repl/step10000-merged \
    /mnt/bn/kinetics-lp-maliva/playground_projects/MolmoAct/checkpoints/libero-all-b100-10k-bs256x4-wconnector-repl/step10000-merged-hf \
    demo_role \
    --precision bf16
"""