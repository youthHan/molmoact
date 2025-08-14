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

## Table of Contents

- [Overview](#overview)
- [Release Notes](#release-notes)
  - [Datasets](#datasets)
  - [Models](#models)
- [License and Use](#license-and-use)
- [Model and Hardware Safety](#model-and-hardware-safety)
- [Citation](#citation)

### Quick Links

| Section | Link |
|---|---|
| Overview | [Go to Overview](#overview) |
| Datasets | [Go to Datasets](#datasets) |
| Models | [Go to Models](#models) |
| License | [Go to License and Use](#license-and-use) |
| Safety | [Go to Model and Hardware Safety](#model-and-hardware-safety) |
| Citation | [Go to Citation](#citation) |

---

## Overview

MolmoAct is a repository for training and using AI2’s open-sourced **Action Reasoning Model** that can reason in space.

> **Note:** Training code, evaluation code, and data processing scripts will be released soon. We’re finalizing them for public release to ensure reproducibility and ease of use.

---

## Release Notes

### Datasets

- **[2025/08/12] 🔥 [Datasets](https://huggingface.co/collections/allenai/molmoact-data-mixture-6897e583e13b6c2cf3ea2b80)** used for our pre-training and mid-training have been released:

| Data                               | Description                                                                                                                                  | Dataset Path                                                             |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| MolmoAct Dataset                   | MolmoAct dataset in LeRobot format. All contents were collected in-house by AI2.                                                            | https://huggingface.co/datasets/allenai/MolmoAct-Dataset                 |
| MolmoAct Pre-training Mixture      | Data mixture for MolmoAct pre-training. Contains a subset of OXE formulated as Action Reasoning data, auxiliary robot data, and web data.   | https://huggingface.co/datasets/allenai/MolmoAct-Pretraining-Mixture     |
| MolmoAct Mid-training Mixture      | Data mixture for MolmoAct mid-training. Contains MolmoAct Dataset formulated as Action Reasoning data.                                      | https://huggingface.co/datasets/allenai/MolmoAct-Midtraining-Mixture     |

### Models

- **[2025/08/12] 🔥 [Models](https://huggingface.co/collections/allenai/molmoact-689697591a3936fba38174d7)** have been released:

| Model                       | Use Case     | Description                                                                                                 | Checkpoint Path                                                |
|----------------------------|--------------|-------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| MolmoAct-7B-D              | Fine-tuning  | Best/demo MolmoAct; adapt to real robots by fine-tuning on your datasets.                                   | https://huggingface.co/allenai/MolmoAct-7B-D-0812              |
| MolmoAct-7B-O              | Fine-tuning  | Most open MolmoAct; adapt to real robots by fine-tuning on your datasets.                                   | https://huggingface.co/allenai/MolmoAct-7B-O-0812              |
| MolmoAct-7B-D-Pretrain     | Inference    | Checkpoint to replicate zero-shot results on SimplerEnv (Google Robot).                                     | https://huggingface.co/allenai/MolmoAct-7B-D-Pretrain-0812     |
| MolmoAct-7B-D-Pretrain-RT-1| Inference    | Checkpoint to replicate RT-1 fine-tuned results on SimplerEnv (Google Robot).                               | https://huggingface.co/allenai/MolmoAct-7B-D-Pretrain-RT-1-0812|

All artifacts used in creating MolmoAct (data, training code, evaluations, intermediate checkpoints) will be made available later to further our commitment to open-source AI development and reproducibility. We will provide links to all artifacts in this repo after release.

---

## License and Use

MolmoAct is licensed under **Apache 2.0** and intended for research and educational use.  
For more information, please see our [Responsible Use Guidelines](https://allenai.org/responsible-use).

---

## Model and Hardware Safety

MolmoAct can display a **visual trace** of its intended actions before execution, enabling proactive auditing and adjustment of behavior. The model’s action space is bounded within the data provided, and compliance is built in to limit excessive force when resistance is detected. Always follow hardware manufacturer guidelines and operate in a safely configured environment.

---

## Citation

```bibtex
@misc{molmoact2025,
      title={MolmoAct: Action Reasoning Models that can Reason in Space}, 
      author={Jason Lee and Jiafei Duan and Haoquan Fang and Yuquan Deng and Shuo Liu and Boyang Li and Bohan Fang and Jieyu Zhang and Yi Ru Wang and Sangho Lee and Winson Han and Wilbert Pumacay and Angelica Wu and Rose Hendrix and Karen Farley and Eli VanderBilt and Ali Farhadi and Dieter Fox and Ranjay Krishna},
      year={2025},
      eprint={2508.07917},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2508.07917}, 
}
