<div align="center">
  <img src="assets/molmoact_logo.svg" alt="MolmoAct Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
  <h1>MolmoAct: Multimodal Open Language Model for Action</h1>
</div>

<p align="center">
  <a href="https://github.com/allenai/MolmoAct/blob/release/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://allenai.org/blog/molmoact">
    <img alt="Blog Post" src="https://img.shields.io/badge/MolmoAct-Blog-F0529C">
  </a>
  <a href="https://arxiv.org/abs/2508.07917">
    <img alt="Paper URL" src="https://img.shields.io/badge/arXiv-2508.07917-red?logo=arxiv">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmoact-689697591a3936fba38174d7">
    <img alt="Model Checkpoints" src="https://img.shields.io/badge/HF-Models-yellow?logo=huggingface">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmoact-data-mixture-6897e583e13b6c2cf3ea2b80">
    <img alt="Datasets" src="https://img.shields.io/badge/HF-Datasets-yellow?logo=huggingface">
  </a>
</p>

---
### Updates
- **[2025/09/06]** 🔥 Code for replicating MolmoAct's training pipeline has been released
- **[2025/08/15]** 🔥 Code for MolmoAct Evaluation on SimplerEnv has been released at  **[allenai/SimplerEnv](https://github.com/allenai/SimplerEnv)**
- **[2025/08/12] 🔥 [Datasets](https://huggingface.co/collections/allenai/molmoact-data-mixture-6897e583e13b6c2cf3ea2b80)** used for our pre-training and mid-training have been released
- **[2025/08/12] 🔥 [Models](https://huggingface.co/collections/allenai/molmoact-689697591a3936fba38174d7)** have been released




## Table of Contents

1. [Overview](#1-overview)  
2. [Release Notes](#2-release-notes)  
 2.1 [Datasets](#21-datasets)  
 2.2 [Models](#22-models)  
3. [Installation](#3-training-wip)  
4. [Training (WIP)](#4-training-wip)  
 4.1 [Data Processing & Fine-tuning (WIP)](#41-data-processing--fine-tuning-post-training-wip)  
 4.2 [Training Replication](#42-training-replication)  
  4.2.1 [Pre-training](#421-pre-training)  
  4.2.2 [Mid-training](#422-mid-training)  
  4.2.3 [Post-training (LIBERO)](#423-post-training-libero)  
5. [Evaluation](#5-evaluation-wip)
5.1 [SimplerEnv](#51-simpler-env)
5.2 [LIBERO](#52-libero)
5.3 [MolmoAct Gradio Portal](#53-molmoact-gradio-portal)
6. [License and Use](#6-license-and-use)  
7. [Model and Hardware Safety](#7-model-and-hardware-safety)  
8. [Citation](#8-citation)  
9. [Contacts](#9-contacts)


---

## 1. Overview

MolmoAct is a repository for training and using AI2’s open-sourced **Action Reasoning Model** that can reason in space.

> **Note:** Training code, evaluation code, and data processing scripts will be released soon. We’re finalizing them for public release to ensure reproducibility and ease of use.

---

## 2. Release Notes

### 2.1 Datasets

| Data                               | Description                                                                                                                                  | Dataset Path                                                             |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| MolmoAct Dataset                   | MolmoAct dataset in LeRobot format. All contents were collected in-house by AI2.                                                            | https://huggingface.co/datasets/allenai/MolmoAct-Dataset                 |
| MolmoAct Pre-training Mixture      | Data mixture for MolmoAct pre-training. Contains a subset of OXE formulated as Action Reasoning data, auxiliary robot data, and web data.   | https://huggingface.co/datasets/allenai/MolmoAct-Pretraining-Mixture     |
| MolmoAct Mid-training Mixture      | Data mixture for MolmoAct mid-training. Contains MolmoAct Dataset formulated as Action Reasoning data.                                      | https://huggingface.co/datasets/allenai/MolmoAct-Midtraining-Mixture     |

### 2.2 Models

| Model                       | Use Case     | Description                                                                                                 | Checkpoint Path                                                |
|----------------------------|--------------|-------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| MolmoAct-7B-D              | Fine-tuning  | Best/demo MolmoAct; adapt to real robots by fine-tuning on your datasets.                                   | https://huggingface.co/allenai/MolmoAct-7B-D-0812              |
| MolmoAct-7B-O              | Fine-tuning  | Most open MolmoAct; adapt to real robots by fine-tuning on your datasets.                                   | https://huggingface.co/allenai/MolmoAct-7B-O-0812              |
| MolmoAct-7B-D-Pretrain     | Inference    | Checkpoint to replicate zero-shot results on SimplerEnv (Google Robot).                                     | https://huggingface.co/allenai/MolmoAct-7B-D-Pretrain-0812     |
| MolmoAct-7B-D-Pretrain-RT-1| Inference    | Checkpoint to replicate RT-1 fine-tuned results on SimplerEnv (Google Robot).                               | https://huggingface.co/allenai/MolmoAct-7B-D-Pretrain-RT-1-0812|
| MolmoAct-7B-D-Captioner | Training Replication    | Checkpoint to replicate MolmoAct training from scratch.                               | https://huggingface.co/allenai/MolmoAct-7B-D-Captioner-0812|

---

## 3. Installation

We provide the `Dockerfile` to build the docker, where we ran all our training experiments on. We strongly recommand to build the same docker on your own and run training on that.

If you want to install environment on your own, first install python 3.11, then install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system. 

Next, in both cases, go to your working molmoact folder, and run:

```bash
git clone https://github.com/allenai/molmoact.git
cd molmo
pip install -e .[all]
```
---

## 4. Training (WIP)

We provide instructions on both how to train your own MolmoAct (WIP) and how to replicate all of our training stages:

### 4.1 Data Processing & Fine-tuning (Post-training) (WIP)
_Content coming soon._
### 4.2 Training Replication

#### Where data is stored
MolmoAct pulls most datasets via **Hugging Face Datasets**; those files go into the Hugging Face cache. A few extra assets are stored under a separate root defined by `MOLMOACT_DATA_DIR`.

Set both paths (example: store everything under `/data/molmoact`):

```bash
export MOLMOACT_DATA_DIR=/data/molmoact
export HF_HOME=/data/molmoact/huggingface
```

> `HF_HOME` controls the Hugging Face cache location. See the official docs on managing the cache [here](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

---

#### Download robot datasets

You can download our robot datasets in many ways, as shown in the following:

**All robot datasets:**
```bash
python3 scripts/download_robot_data.py all --n_proc 16
```

**Specific training stage:**
```bash
python3 scripts/download_robot_data.py <stage> --n_proc 16
```
Use one of: `pretrain`, `midtrain`, `libero`.

**Single robot dataset class by name:**
```bash
python3 scripts/download_robot_data.py MolmoActDatasetHomePrimary --n_proc 16
```
> All robot dataset class names are listed at the end of `olmo/data/robot_datasets.py`.

---

#### Download Molmo (Multimodal Web) data
These are the **Multimodal Web Data** used during MolmoAct pre-training.

**All web datasets (after setting `MOLMOACT_DATA_DIR` and `HF_HOME`):**
```bash
python3 scripts/download_data.py all --n_proc 16
```

**Single web dataset (example):**
```bash
python3 scripts/download_data.py ChartQa --n_proc 16
```

---

#### Notes & tips
- **Pixmo** datasets fetch images from URLs. The script does this automatically but may take a long time; a full fresh download can take **up to a day**.
- `--n_proc` controls parallelism. More processes can speed things up but also increase the chance of **rate limiting**.
- Downloads are **resumable** if you cancel or hit an error.
- Some datasets (**InfoQa**, **Scene-Text**) require **manual downloads**. The scripts will raise an error if those files are missing.
- The **Android control** dataset needs extra dependencies because it parses original **TFRecords**.
- We recommend ensuring the data is downloaded and then using the environment variable `HF_DATASETS_OFFLINE=1` during training to ensure the nodes don't flood HF with requests as they all initialize and then potentially get rate limited.


#### 4.2.1 Pre-training

**Command**
```bash
WANDB_API_KEY=<your_wandb_api_key> torchrun \
    --nnodes=32 --nproc-per-node=8 \
    --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    launch_scripts/train_multitask_model.py \
    molmoact-pretrain allenai/MolmoAct-7B-D-Captioner-0812 \
    --wandb.name=<name> --wandb.entity=<entity> --wandb.project=<project>  \
    --save_folder=checkpoints/<exp_name> \
    --save_overwrite \
    --duration 100000 \
    --ft_embedding all \
    --depth_tokens \
    --global_batch_size 512 \
    --lr_connector 1e-5 \
    --lr_vit 1e-5 \
    --lr_llm 2e-5 \
    --save_interval 20000 \
    --save_num_checkpoints_to_keep 5 \
    --save_final_unsharded_checkpoint
```

**Fill these placeholders**
- `WANDB_API_KEY=<your_wandb_api_key>` → your Weights & Biases (W&B) API key.
- `--wandb.name=<name> --wandb.entity=<entity> --wandb.project=<project>` → your Weights & Biases (W&B) run info.
- `--save_folder=checkpoints/<exp_name>` → folder name for checkpoints (use a unique experiment name).

**W&B logging**
- Offline logging: `WANDB_MODE=offline`.
- Turn off wandb: replace `--wandb.name=<name> --wandb.entity=<entity> --wandb.project=<project>` with `--wandb=null`.

**Checkpoints & formats**
- By default **all** intermediate checkpoints are **sharded**; only the **final** checkpoint is also saved **unsharded** (`--save_final_unsharded_checkpoint`).
- To save **unsharded copies for every checkpoint**, add: `--save_intermediate_unsharded_checkpoint`.

**Cluster launch variables**
- Set these per your cluster/launcher:  
  `--node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}"`.

**Notes**
- Avoid `--pin_memory` for large datasets; it can cause OOM during loading.


---

#### 4.2.2 Mid-training

**Command**
```bash
WANDB_API_KEY=<your_wandb_api_key> torchrun --nnodes=16 --nproc-per-node=8 \
    --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    launch_scripts/train_multitask_model.py \
    molmoact-midtrain allenai/MolmoAct-7B-D-Pretrain-0812 \
    --wandb.name=<name> --wandb.entity=<entity> --wandb.project=<project>  \
    --save_folder=checkpoints/<exp_name> \
    --save_overwrite \
    --duration 50000 \
    --ft_embedding all \
    --depth_tokens \
    --global_batch_size 256 \
    --lr_connector 5e-6 \
    --lr_vit 5e-6 \
    --lr_llm 1e-5 \
    --save_interval 10000 \
    --save_num_checkpoints_to_keep 5 \
    --save_final_unsharded_checkpoint \
    --max_images 2
```

**What’s different from pre-training**
- Base checkpoint: `allenai/MolmoAct-7B-D-Pretrain-0812`.
- Hyperparameters change (shorter `--duration`, smaller `--global_batch_size`, lower LRs).
- `--max_images 2` indicates each training example uses **two images**.
- All other setup (W&B, saving, cluster vars) follows the **pre-training** instructions.


---

#### 4.2.3 Post-training (LIBERO)

**Command**
```bash
WANDB_API_KEY=<your_wandb_api_key> torchrun --nnodes=8 --nproc-per-node=8 \
    --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    launch_scripts/train_multitask_model.py \
    libero-<task_suite> allenai/MolmoAct-7B-D-0812 \
    --wandb.name=<name> --wandb.entity=<entity> --wandb.project=<project>  \
    --save_folder=checkpoints/<exp_name> \
    --save_overwrite \
    --duration <steps> \
    --ft_embedding all \
    --depth_tokens \
    --global_batch_size 128 \
    --lr_connector 5e-4 \
    --lr_vit 5e-4 \
    --lr_llm 5e-4 \
    --save_interval 10000 \
    --save_num_checkpoints_to_keep 5 \
    --save_final_unsharded_checkpoint \
    --max_images 2 \
    --lora_enable --lora_rank 32 --lora_alpha 16 --lora_dropout 0.0 \
    --img_aug
```

**What’s different here**
- Base checkpoint: `allenai/MolmoAct-7B-D-0812`.
- Uses **LoRA** fine-tuning (`--lora_enable ...`) and **image augmentation** (`--img_aug`).
- `--max_images 2` again indicates two images per input.
- Choose `--duration <steps>` based on the **LIBERO task suite**.

**Choose `<task_suite>` and `<steps>`**
| `<task_suite>` | `<steps>` |
|---|---|
| spatial | 50000 |
| object  | 50000 |
| goal    | 40000 |
| long    | 80000 |

**Reminder**
- Follow the **pre-training** notes for W&B setup, checkpointing behavior, and cluster launch variables; those apply here as well.

## 5. Evaluation

### 5.1 Simpler-Env

We release the SimplerEnv evaluation code for MolmoAct at [allenai/SimplerEnv](https://github.com/allenai/SimplerEnv). Please first install the dependencies for SimplerEnv Evaluation environment following [allenai/SimplerEnv](https://github.com/allenai/SimplerEnv) and dependencies for [MolmoAct Inference Setup](https://github.com/allenai/SimplerEnv?tab=readme-ov-file#molmoact-inference-setup). After installing all the dependencies, evaluation scripts are located at:


```bash
# under the project dir of SimplerEnv/
bash scripts/molmoact_pick_coke_can_visual_matching.sh
bash scripts/molmoact_pick_coke_can_variant_agg.sh
bash scripts/molmoact_move_near_visual_matching.sh
bash scripts/molmoact_move_near_variant_agg.sh
bash scripts/molmoact_drawer_visual_matching.sh
bash scripts/molmoact_drawer_variant_agg.sh
```



### 5.2 LIBERO

```bash
# under the project dir of molmoact/
cd experiments/LIBERO
pip install -e .
pip install einops torchvision accelerate
pip install transformers==4.52.1
pip install vllm==0.8.5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd ../libero

# to replicate molmoact results with vllm
python run_libero_eval_vllm.py --task spatial --checkpoint allenai/MolmoAct-7B-D-LIBERO-Spatial-0812
python run_libero_eval_vllm.py --task object --checkpoint allenai/MolmoAct-7B-D-LIBERO-Object-0812
python run_libero_eval_vllm.py --task goal --checkpoint allenai/MolmoAct-7B-D-LIBERO-Goal-0812
python run_libero_eval_vllm.py --task 10 --checkpoint allenai/MolmoAct-7B-D-LIBERO-Long-0812

# we also provide the code to run libero with only huggingface
python run_libero_eval.py --task spatial --checkpoint allenai/MolmoAct-7B-D-LIBERO-Spatial-0812
python run_libero_eval.py --task object --checkpoint allenai/MolmoAct-7B-D-LIBERO-Object-0812
python run_libero_eval.py --task goal --checkpoint allenai/MolmoAct-7B-D-LIBERO-Goal-0812
python run_libero_eval.py --task 10 --checkpoint allenai/MolmoAct-7B-D-LIBERO-Long-0812
```

### 5.3 MolmoAct Gradio Portal

Launch the interactive portal to explore MolmoAct predictions on your own images.

```bash
pip install gradio  # or pip install -e .[serve]
python scripts/gradio_portal.py --checkpoint <hf_repo_or_local_path>
```

Add `--device cuda` (default when available) to run on GPU or `--device cpu` to stay on CPU. The portal keeps a running conversation with the model, accepts supplementary images on later turns, and always renders predicted traces on the first front-view image.

**Key helpers (`scripts/gradio_portal.py`):**
- `_load_images` standardises Gradio uploads into RGB PIL images before they reach the processor.
- `_max_images_from_config` queries the checkpoint’s multimodal config so the UI enforces the model’s image budget.
- `_gather_images_from_messages`, `_latest_user_images`, `_first_user_image` recover all stored frames, the newest batch, and the canonical front-view used for overlay rendering.
- `_normalize_trace` and `_draw_traces_on_images` scale model traces, pick colours, and paint markers/lines over the chosen image.

**Generation loop:**
1. `infer` (triggered by the **Generate** button) validates the new text/images against the limit and appends the user turn to the running message state.
2. `processor.apply_chat_template` formats the multi-round transcript; `processor(**kwargs)` tokenises text and bundles the accumulated images.
3. Inputs are moved to the target device via `_cast_for_device`, then `model.generate` produces the next response.
4. The decoded assistant reply updates the chat log and is parsed through `parse_depth/parse_trace/parse_action`. Overlays are redrawn on the stored front-view image.
5. `gr.State`, the chatbot widget, JSON panes, and gallery refresh in one pass; a chained `.then` clears the file upload area for the next turn.

```
User Uploads/Text
        │
        ▼
_load_images ──► message state update ──► apply_chat_template
                                          │
                                          ▼
                                   processor(**)
                                          │
                                          ▼
                                  model.generate
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
              batch_decode                             parse_depth/trace/action
                     │                                         │
                     ▼                                         │
             chat history update ◄──── front-view selector ────┘
                     │
                     ▼
              _draw_traces_on_images
                     │
                     ▼
              Gradio widgets refresh
```

**Usage tips:**
- Upload your primary front-view image on the first turn; later image uploads act as auxiliary observations for the language model.
- The gallery is capped to a single column (320 px high) so trace previews stay compact during multi-round chats.
- Use the **Clear Conversation** button to reset both the dialogue and the image budget.

## 6. License and Use

MolmoAct is licensed under **Apache 2.0** and intended for research and educational use.  
For more information, please see our [Responsible Use Guidelines](https://allenai.org/responsible-use).

---

## 7. Model and Hardware Safety

MolmoAct can display a **visual reasoning trace** of its intended actions before execution, enabling proactive auditing and adjustment of behavior. The model’s action space is bounded within the data provided, and compliance is built in to limit excessive force when resistance is detected. Always follow hardware manufacturer guidelines and operate in a safely configured environment.

---

## 8. Citation

```bibtex
@misc{molmoact2025,
      title={MolmoAct: Action Reasoning Models that can Reason in Space}, 
      author={Jason Lee and Jiafei Duan and Haoquan Fang and Yuquan Deng and Shuo Liu and Boyang Li and Bohan Fang and Jieyu Zhang and Yi Ru Wang and Sangho Lee and Winson Han and Wilbert Pumacay and Angelica Wu and Rose Hendrix and Karen Farley and Eli VanderBilt and Ali Farhadi and Dieter Fox and Ranjay Krishna},
      year={2025},
      eprint={2508.07917},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2508.07917}
}
```

---

## 9. Contacts

For questions, collaborations, or support, please contact with:

```
{haoquanf,jasonl,jiafeid}@allenai.org 
```

Found a bug or have a feature request? Please open a [GitHub issue](https://github.com/allenai/MolmoAct/issues).
