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
    <img alt="Model Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-yellow">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmoact-data-mixture-6897e583e13b6c2cf3ea2b80">
    <img alt="Datasets" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Datasets-yellow">
  </a>
</p>

---
### Updates
- **[2025/08/15]** 🔥 Code for MolmoAct Evaluation on SimplerEnv has been released at  **[allenai/SimplerEnv](https://github.com/allenai/SimplerEnv)**
- **[2025/08/12] 🔥 [Datasets](https://huggingface.co/collections/allenai/molmoact-data-mixture-6897e583e13b6c2cf3ea2b80)** used for our pre-training and mid-training have been released
- **[2025/08/12] 🔥 [Models](https://huggingface.co/collections/allenai/molmoact-689697591a3936fba38174d7)** have been released




## Table of Contents

1. [Overview](#1-overview)  
2. [Release Notes](#2-release-notes)  
 2.1 [Datasets](#21-datasets)  
 2.2 [Models](#22-models)  
3. [Training (WIP)](#3-training-wip)  
 3.1 [Data Processing & Fine-tuning](#31-data-processing--fine-tuning)  
 3.2 [Pre-training](#32-pre-training)  
 3.3 [Mid-training](#33-mid-training)  
4. [Evaluation (WIP)](#4-evaluation-wip)  
 4.1 [SimplerEnv](#41-simpler-env)  
 4.2 [LIBERO Evaluation](#42-libero-evaluation)  
 4.3 [Real-world Evaluation](#43-real-world-evaluation)  
5. [License and Use](#5-license-and-use)  
6. [Model and Hardware Safety](#6-model-and-hardware-safety)  
7. [Citation](#7-citation)  
8. [Contacts](#8-contacts)

### Quick Links

| Section | Link |
|---|---|
| Overview | [#1-overview](#1-overview) |
| Datasets | [#21-datasets](#21-datasets) |
| Models | [#22-models](#22-models) |
| Training | [#3-training-wip](#3-training-wip) |
| Evaluation | [#4-evaluation-wip](#4-evaluation-wip) |
| License | [#5-license-and-use](#5-license-and-use) |
| Safety | [#6-model-and-hardware-safety](#6-model-and-hardware-safety) |
| Citation | [#7-citation](#7-citation) |

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

---

## 3. Training (WIP)

### 3.1 Data Processing & Fine-tuning
_Content coming soon._

### 3.2 Pre-training
_Content coming soon._

### 3.3 Mid-training
_Content coming soon._

---

## 4. Evaluation (WIP)

### 4.1 Simpler-Env

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



### 4.2 LIBERO Evaluation
_Content coming soon._

### 4.3 Real-world Evaluation
_Content coming soon._

---

## 5. License and Use

MolmoAct is licensed under **Apache 2.0** and intended for research and educational use.  
For more information, please see our [Responsible Use Guidelines](https://allenai.org/responsible-use).

---

## 6. Model and Hardware Safety

MolmoAct can display a **visual reasoning trace** of its intended actions before execution, enabling proactive auditing and adjustment of behavior. The model’s action space is bounded within the data provided, and compliance is built in to limit excessive force when resistance is detected. Always follow hardware manufacturer guidelines and operate in a safely configured environment.

---

## 7. Citation

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

## 8. Contacts

For questions, collaborations, or support, please contact with:

```
{haoquanf,jasonl,jiafeid}@allenai.org 
```

Found a bug or have a feature request? Please open a [GitHub issue](https://github.com/allenai/MolmoAct/issues).
