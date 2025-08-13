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

MolmoAct is a repository for training and using Ai2's open-sourced Action Reasoning Model that can reason in space.

**Note: Training code, evaluation code, and Data Processing scripts will be released soon. We are finalizing them for public release to ensure ease of reproducibility and clarity.**


## Release Notes

- [2025/08/12] **🔥 [Datasets](https://huggingface.co/collections/allenai/molmoact-data-mixture-6897e583e13b6c2cf3ea2b80)** used for our pre-training and mid-training has been released, which consists of:

| Data                                  | Description                                                                                                                                                             | Dataset Path                                                              |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| MolmoAct Dataset                      | This dataset contains MolmoAct Dataset in Lerobot format. All contents in this dataset were collected in-house by Ai2.                                                  | https://huggingface.co/datasets/allenai/MolmoAct-Dataset                  |
| MolmoaAct Pre-training Mixture        | Data Mixture used for MolmoAct Pretraining. Contains a subset of OXE formulated as Action Reasoning Data along with auxiliary robot data and Multimodal Web data        | https://huggingface.co/datasets/allenai/MolmoAct-Pretraining-Mixture      |
| MolmoAct Mid-training Mixture         | Data Mixture used for MolmoAct Midtraining. Contains MolmoAct Dataset formulated as Action Reasoning Data                                                               | https://huggingface.co/datasets/allenai/MolmoAct-Midtraining-Mixture      |




- [2025/08/12] **🔥 [Models](https://huggingface.co/collections/allenai/molmoact-689697591a3936fba38174d7)** has been released, which consists of:


| Model                       | Use Case          | Description                                                                                                  | Checkpoint Path                                                 |
| --------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| MolmoAct-7B-D               | Fine-Tuning       | Our best and demo version of MolmoAct that can be adapted to real world by fine-tuning on your own datasets  | https://huggingface.co/allenai/MolmoAct-7B-D-0812               |
| MolmoAct-7B-O               | Fine-Tuning       | Our most open version of MolmoAct that can be adapt to real world by fine-tuneing on your own datasets       | https://huggingface.co/allenai/MolmoAct-7B-O-0812               |
| MolmoAct-7B-D-Pretrain      | Inference         | MolmoAct checkpoint used to replicate zero-shot results on SimplerEnv (Google Robot)                         | https://huggingface.co/allenai/MolmoAct-7B-D-Pretrain-0812      |
| MolmoAct-7B-D-Pretrain-RT-1 | Inference         | MolmoAct checkpoint used to replicate fine-tuned results on SimplerEnv (Google Robot)                        | https://huggingface.co/allenai/MolmoAct-7B-D-Pretrain-RT-1-0812 |



All artifacts used in creating MolmoAct (data, training code, evaluations, intermediate checkpoints) will be made available at a later date, furthering our commitment to open-source AI development and reproducibility. We will provied link to all artifacts in this repo after release.

## License and Use

MolmoAct is licensed under Apache 2.0. It is intended for research and educational use.
For more information, please see our [Responsible Use Guidelines](https://allenai.org/responsible-use).


## Model and Hardware Safety
MolmoAct offers the ability to inspect a visual trace of its intended actions in space before they occur, allowing users to ensure safe behavior by proactively auditing and adjusting the actions of any hardware acting under the model’s instructions. MolmoAct’s action space is bounded within the data provided, and compliance is built into the model to prevent excessive force when resistance is detected. Please follow the hardware manufacturer’s guidelines when using this model with a robot and perform all operations in a safely configured environment.


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
```

