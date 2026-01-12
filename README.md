# Awesome RGB-T Salient Object Detection

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
![Papers](https://img.shields.io/badge/Papers-100%2B-blue)
![Last Update](https://img.shields.io/badge/Last%20Update-2026--01-orange)

A curated and continuously updated collection of **RGB–Thermal Salient Object Detection (RGB-T SOD)** methods, datasets, and related resources.

RGB-T SOD exploits the complementary properties of **visible (RGB)** and **thermal infrared (T)** modalities to achieve robust saliency perception under challenging conditions such as **low illumination, camouflage, occlusion, adverse weather, and background clutter**.

---

## 📌 Contents

* [Papers](#-papers)

  * [Machine Learning-based Methods](#machine-learning-based-methods)
  * [Deep Learning-based Methods](#deep-learning-based-methods)
  * [Other Paradigms](#other-paradigms)
* [Datasets](#-datasets)
* [Evaluation Metrics](#-evaluation-metrics)
* [Related Surveys](#-related-surveys)
* [Citation](#-citation)

---

## 📚 Papers

### Machine Learning-based Methods

| Year | Method  | Title                                                                                   | Venue | Resources (Paper / Code) |
| ---- | ------- | --------------------------------------------------------------------------------------- | ----- | ------------------------ |
| 2018 | MTMR    | RGB-T saliency detection benchmark: Dataset, baselines, analysis and a novel approach   | IGTA  | [Paper](https://link.springer.com/chapter/10.1007/978-981-13-1702-6_36) / [代码](https://github.com/lz118/RGBT-Salient-Object-Detection/blob/master/Readme.md)     |
| 2019 | M3S-NIR | M3S-NIR: Multi-Modal Multi-Scale Noise-Insensitive Ranking for RGB-T Saliency Detection | MIPR  | [Paper]() / [代码]()     |
| 2019 | SGDL    | RGB-T Image Saliency Detection via Collaborative Graph Learning                         | TMM   | [Paper]() / [代码]()     |
| 2020 | LTCR    | RGB-T Saliency Detection via Low-Rank Tensor Learning and Unified Collaborative Ranking | SPL   | [Paper]() / [代码]()     |
| 2022 | MGFL    | Multi-Graph Fusion and Learning for RGBT Image Saliency Detection                       | TCSVT | [Paper]() / [代码]()     |

---

### Deep Learning-based Methods

> **Note**: Backbones are listed as reported in the original papers. This list is intended to be **exhaustive** based on the current repository content and will be continuously updated.

| Year | Method     | Title                                                                                       | Venue | Backbone             | Resources (Paper / Code)                                                    |
| ---- | ---------- | ------------------------------------------------------------------------------------------- | ----- | -------------------- | --------------------------------------------------------------------------- |
| 2017 | –          | Learning Multiscale Deep Features and SVM Regressors for Adaptive RGB-T Saliency Detection  | ISCID | VGG16                | [Paper]() / [代码]()                                                        |
| 2020 | FMCF       | RGB-T Salient Object Detection via Fusing Multi-Level CNN Features                          | TIP   | VGG16                | [Paper]() / [代码]()                                                        |
| 2020 | –          | Multi-Spectral Salient Object Detection by Adversarial Domain Adaptation                    | AAAI  | VGG16                | [Paper]() / [代码](https://tsllb.github.io/MultiSOD.html)                   |
| 2020 | –          | Deep Domain Adaptation Based Multi-spectral Salient Object Detection                        | TMM   | VGG16                | [Paper]() / [代码](https://tsllb.github.io/MultiSOD.html)                   |
| 2021 | TSFNet     | TSFNet: Two-Stage Fusion Network for RGB-T Salient Object Detection                         | SPL   | ResNet-34            | [Paper]() / [代码]()                                                        |
| 2021 | FFNet      | Revisiting Feature Fusion for RGB-T Salient Object Detection                                | TCSVT | VGG16                | [Paper]() / [代码]()                                                        |
| 2021 | MIDD       | Multi-Interactive Dual-Decoder for RGB-Thermal Salient Object Detection                     | TIP   | VGG16                | [Paper]() / [代码](https://github.com/lz118/Multi-interactive-Dual-decoder) |
| 2021 | MMNet      | Unified Information Fusion Network for Multi-Modal RGB-D and RGB-T Salient Object Detection | TCSVT | VGG19 / Res2Net-50   | [Paper]() / [代码]()                                                        |
| 2021 | –          | Salient Target Detection in RGB-T Image based on Multi-level Semantic Information           | CYBER | VGG16                | [Paper]() / [代码]()                                                        |
| 2022 | CGFNet     | CGFNet: Cross-Guided Fusion Network for RGB-T Salient Object Detection                      | TCSVT | VGG16                | [Paper]() / [代码](https://github.com/wangjie0825/CGFNet.git)               |
| 2022 | CGMDRNet   | Cross-Guided Modality Difference Reduction Network for RGB-T Salient Object Detection       | TCSVT | Res2Net-50           | [Paper]() / [代码]()                                                        |
| 2022 | CCFENet    | Cross-Collaborative Fusion-Encoder Network for Robust RGB-Thermal Salient Object Detection  | TCSVT | ResNet-34            | [Paper]() / [代码](https://git.openi.org.cn/OpenVision/CCFENet)             |
| 2022 | ECFFNet    | Effective and Consistent Feature Fusion Network for RGB-T Salient Object Detection          | TCSVT | ResNet-34            | [Paper]() / [代码]()                                                        |
| 2022 | CSRNet     | Efficient Context-Guided Stacked Refinement Network for RGB-T Salient Object Detection      | TCSVT | ESPNetv2             | [Paper]() / [代码](https://github.com/huofushuo/CSRNet)                     |
| 2022 | SwinNet    | Swin Transformer Drives Edge-Aware RGB-D and RGB-T SOD                                      | TCSVT | Swin                 | [Paper]() / [代码](https://github.com/liuzywen/SwinNet)                     |
| 2022 | APNet      | Adversarial Learning Assistance and Importance Fusion Network for All-Day RGB-T SOD         | TETCI | VGG16                | [Paper]() / [代码]()                                                        |
| 2022 | OSRNet     | One-Stream Semantic-Guided Refinement Network for RGB-T SOD                                 | TIM   | VGG16 / ResNet-50    | [Paper]() / [代码](https://github.com/huofushuo/OSRNet)                     |
| 2022 | DCNet      | Weakly Alignment-Free RGBT Salient Object Detection With Deep Correlation Network           | TIP   | VGG16                | [Paper]() / [代码]()                                                        |
| 2022 | CFRNet     | CNN Feature and Result Fusion for RGB-T Salient Object Detection                            | AI    | VGG16                | [Paper]() / [代码]()                                                        |
| 2022 | MIA-DPD    | Multi-modal Interactive Attention and Dual Progressive Decoding Network                     | NC    | ResNet-50            | [Paper]() / [代码](https://github.com/Liangyh18/MIA_DPD)                    |
| 2022 | SwinMCNet  | Mirror Complementary Transformer Network for RGB-T SOD                                      | arXiv | Swin-B               | [Paper]() / [代码](https://github.com/jxr326/SwinMCNet)                     |
| 2023 | HRTransNet | HRFormer-Driven Two-Modality Salient Object Detection                                       | TCSVT | HRFormer / ResNet18  | [Paper]() / [代码](https://github.com/liuzywen/HRTransNet)                  |
| 2023 | WaveNet    | Wavelet Network with Knowledge Distillation for RGB-T SOD                                   | TIP   | Wave-MLP             | [Paper]() / [代码](https://github.com/nowander/WaveNet)                     |
| 2023 | LSNet      | Lightweight Spatial Boosting Network for RGB-Thermal SOD                                    | TIP   | MobileNetV2          | [Paper]() / [代码](https://github.com/zyrant/LSNet)                         |
| 2023 | TAGFNet    | Thermal-aware Guided Early Fusion Network                                                   | EAAI  | VGG16                | [Paper]() / [代码](https://github.com/VDT-2048/TAGFNet)                     |
| 2024 | TCINet     | Transformer-Based Cross-Modal Integration Network for RGB-T SOD                             | TCE   | Swin                 | [Paper]() / [代码](https://github.com/lvchengtao/TCINet)                    |
| 2024 | MSEDNet    | Multi-scale Fusion and Edge-supervised Network for RGB-T SOD                                | NN    | ResNet-34/50/101/152 | [Paper]() / [代码](https://github.com/Zhou-wy/MSEDNet)                      |
| 2025 | CONTRINET  | Divide-and-Conquer: Confluent Triple-Flow Network for RGB-T SOD                             | TPAMI | VGG / Res2Net / Swin | [Paper]() / [代码](https://github.com/CSer-Tang-hao/ConTriNet_RGBT-SOD)     |
| 2025 | Samba      | A Unified Mamba-based Framework for General SOD                                             | CVPR  | VMamba               | [Paper]() / [代码](https://github.com/Jia-hao999/Samba)                     |

------|--------|-------|-------|----------|-----------|
| 2017 | – | Learning Multiscale Deep Features and SVM Regressors for Adaptive RGB-T Saliency Detection | ISCID | VGG16 | – |
| 2020 | FMCF | RGB-T Salient Object Detection via Fusing Multi-Level CNN Features | TIP | VGG16 | – |
| 2020 | – | Multi-Spectral Salient Object Detection by Adversarial Domain Adaptation | AAAI | VGG16 | [Paper]() / [代码]() |
| 2020 | – | Deep Domain Adaptation Based Multi-spectral Salient Object Detection | TMM | VGG16 | [Paper]() / [代码]() |
| 2021 | TSFNet | TSFNet: Two-Stage Fusion Network for RGB-T Salient Object Detection | SPL | ResNet-34 | – |
| 2021 | FFNet | Revisiting Feature Fusion for RGB-T Salient Object Detection | TCSVT | VGG16 | – |
| 2021 | MIDD | Multi-Interactive Dual-Decoder for RGB-Thermal Salient Object Detection | TIP | VGG16 | [Paper]() / [代码]() |
| 2021 | MMNet | Unified Information Fusion Network for Multi-Modal RGB-D and RGB-T Salient Object Detection | TCSVT | VGG19 / Res2Net-50 | – |
| 2022 | CGFNet | Cross-Guided Fusion Network for RGB-T Salient Object Detection | TCSVT | VGG16 | [Paper]() / [代码]() |
| 2022 | CGMDRNet | Cross-Guided Modality Difference Reduction Network for RGB-T Salient Object Detection | TCSVT | Res2Net-50 | – |
| 2022 | CSRNet | Efficient Context-Guided Stacked Refinement Network for RGB-T Salient Object Detection | TCSVT | ESPNetv2 | [Paper]() / [代码]() |
| 2022 | SwinNet | Swin Transformer Drives Edge-Aware RGB-D and RGB-T SOD | TCSVT | Swin | [Paper]() / [代码]() |
| 2023 | caver | Cross-Modal View-Mixed Transformer for Bi-Modal Salient Object Detection | TIP | ResNet50/101 | [Paper]() / [代码]() |
| 2023 | WaveNet | Wavelet Network with Knowledge Distillation for RGB-T SOD | TIP | Wave-MLP | [Paper]() / [代码]() |
| 2024 | TCINet | Transformer-Based Cross-Modal Integration Network for RGB-T SOD | TCE | Swin | [Paper]() / [代码]() |
| 2025 | CONTRINET | Divide-and-Conquer: Confluent Triple-Flow Network for RGB-T SOD | TPAMI | VGG / Res2Net / Swin | [Paper]() / [代码]() |
| 2025 | Samba | Samba: A Unified Mamba-based Framework for General SOD | CVPR | VMamba | [Paper]() / [代码]() |

---

### Other Paradigms

#### Vision Foundation Model-based Methods

| Year | Method  | Title                                              | Venue | Backbone   | Resources |
| ---- | ------- | -------------------------------------------------- | ----- | ---------- | --------- |
| 2025 | KAN-SAM | Kolmogorov–Arnold Network Guided SAM for RGB-T SOD | arXiv | SAM2 + KAN | –         |
| 2025 | HyPSAM  | Hybrid Prompt-driven SAM for RGB-T SOD             | TCSVT | Swin-V2    | –         |

#### Diffusion Model-based Methods

| Year | Method  | Title                                         | Venue  | Backbone               | Resources            |
| ---- | ------- | --------------------------------------------- | ------ | ---------------------- | -------------------- |
| 2025 | diffSOD | Unified Diffusion Model for Multi-Modal SOD   | ICASSP | Diffusion + De-ViT     | [Paper]() / [代码]() |
| 2025 | DiMSOD  | Diffusion-Based Framework for Multi-Modal SOD | AAAI   | Stable Diffusion + PVT | –                    |

---

## 🗂 Datasets

| Dataset | Modalities | Size | Scene            |
| ------- | ---------- | ---- | ---------------- |
| VT5000  | RGB-T      | 5000 | Indoor / Outdoor |
| VT1000  | RGB-T      | 1000 | Outdoor          |

---

## 📐 Evaluation Metrics

Commonly used evaluation metrics in RGB-T SOD include:

* **Precision–Recall (PR) Curve**
* **F-measure / Fβ-score**
* **Mean Absolute Error (MAE)**
* **S-measure (Structure-measure)**
* **E-measure (Enhanced-alignment measure)**

---

## 📖 Related Surveys

* RGB-D and RGB-T Salient Object Detection: A Comprehensive Survey
* Multi-Modal Saliency Detection: Datasets, Methods, and Challenges

---

## 📎 Citation

If you find this repository useful, please consider citing or starring ⭐ it.

```bibtex
@misc{rgbt_sod_awesome,
  title  = {Awesome RGB-T Salient Object Detection},
  author = {Community Contributors},
  year   = {2025}
}
```
