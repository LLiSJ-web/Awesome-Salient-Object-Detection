# Awesome RGB-T Salient Object Detection

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
![Last Update](https://img.shields.io/badge/Last%20Update-2026--01-orange)

A curated and continuously updated collection of **RGB–Thermal Salient Object Detection (RGB-T SOD)** methods, datasets, and related resources.

RGB-T SOD exploits the complementary properties of **visible (RGB)** and **thermal infrared (T)** modalities to achieve robust saliency detection under challenging conditions such as **low illumination, camouflage, occlusion, adverse weather, and background clutter**.

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

| Year | Method  | Title                                                                                   | Pub. | Resources|
| ---- | ------- | --------------------------------------------------------------------------------------- | ----- | ------------------------ |
| 2018 | MTMR    | RGB-T saliency detection benchmark: Dataset, baselines, analysis and a novel approach   | IGTA  | [Paper](https://link.springer.com/chapter/10.1007/978-981-13-1702-6_36) / [代码](https://github.com/lz118/RGBT-Salient-Object-Detection/blob/master/Readme.md)     |
| 2019 | M3S-NIR | M3S-NIR: Multi-Modal Multi-Scale Noise-Insensitive Ranking for RGB-T Saliency Detection | MIPR  | [Paper](https://ieeexplore.ieee.org/document/8695412) / [代码](https://github.com/lz118/RGBT-Salient-Object-Detection/tree/master/Code/M3S-NIR)     |
| 2019 | SGDL    | RGB-T Image Saliency Detection via Collaborative Graph Learning                         | TMM   | [Paper](https://ieeexplore.ieee.org/document/8744296) / [代码](https://github.com/lz118/RGBT-Salient-Object-Detection)     |
| 2020 | LTCR    | RGB-T Saliency Detection via Low-Rank Tensor Learning and Unified Collaborative Ranking | SPL   | [Paper](https://ieeexplore.ieee.org/document/9184226?signout=success) / [代码](https://github.com/huanglm-me/LTCR)     |
| 2022 | MGFL    | Multi-Graph Fusion and Learning for RGBT Image Saliency Detection                       | TCSVT | [Paper](https://ieeexplore.ieee.org/document/9389777) / [代码](https://github.com/lmhuang-me/RGBT_MGFL)     |

---

### Deep Learning-based Methods

> **Note**: Backbones are listed as reported in the original papers. This list is intended to be **exhaustive** based on the current repository content and will be continuously updated.


| Year | Method         | Title                                                                                                                                | Pub    | Backbone                 | Resources                                                                      |
| ---- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------ | ------------------------ | ------------------------------------------------------------------------------ |
| 2017 | -              | Learning Multiscale Deep Features and SVM Regressors for Adaptive RGB-T Saliency Detection                                           | ISCID  | VGG16                    | [Paper](https://ieeexplore.ieee.org/document/8275796)                                                                      |
| 2020 | FMCF           | RGB-T Salient Object Detection via Fusing Multi-Level CNN Features                                                                   | TIP    | VGG16                    | [Paper](https://ieeexplore.ieee.org/document/8935533)                                                                      |
| 2020 | -              | Multi-Spectral Salient Object Detection by Adversarial Domain Adaptation                                                             | AAAI   | VGG16                    | [Paper](https://scholarworks.utrgv.edu/cs_fac/73/) / [代码](https://tsllb.github.io/MultiSOD.html)                        |
| 2020 | -              | Deep Domain Adaptation Based Multi-spectral Salient Object Detection                                                                 | TMM    | VGG16                    | [Paper](https://ieeexplore.ieee.org/document/9308922) / [代码](https://tsllb.github.io/MultiSOD.html)                        |
| 2021 | TSFNet         | TSFNet: Two-Stage Fusion Network for RGB-T Salient Object Detection                                                                  | SPL    | ResNet-34                | [Paper](https://ieeexplore.ieee.org/document/9508840)                                                                      |
| 2021 | FFNet          | Revisiting Feature Fusion for RGB-T Salient Object Detection                                                                         | TCSVT  | VGG16                    | [Paper](https://ieeexplore.ieee.org/document/9161021)                                                                      |
| 2021 | MIDD           | Multi-Interactive Dual-Decoder for RGB-Thermal Salient Object Detection                                                              | TIP    | VGG16                    | [Paper](https://ieeexplore.ieee.org/document/9454273) / [代码](https://github.com/lz118/Multi-interactive-Dual-decoder)      |
| 2021 | MMNet          | Unified Information Fusion Network for Multi-Modal RGB-D and RGB-T Salient Object Detection                                          | TCSVT  | VGG19 / Res2Net-50       | [Paper](https://ieeexplore.ieee.org/document/9439490)                                                                      |
| 2021 | -              | Salient Target Detection in RGB-T Image based on Multi-level Semantic Information                                                    | CYBER  | VGG16                    | [Paper](https://ieeexplore.ieee.org/document/9588280)                                                                      |
| 2022 | CGFNet         | CGFNet: Cross-Guided Fusion Network for RGB-T Salient Object Detection                                                               | TCSVT  | VGG16                    | [Paper](https://ieeexplore.ieee.org/document/9493207) / [代码](https://github.com/wangjie0825/CGFNet.git)                    |
| 2022 | CGMDRNet       | CGMDRNet: Cross-Guided Modality Difference Reduction Network for RGB-T Salient Object Detection                                      | TCSVT  | Res2Net-50               | [Paper](https://ieeexplore.ieee.org/document/9756028)                                                                      |
| 2022 | CCFENet        | Cross-Collaborative Fusion-Encoder Network for Robust RGB-Thermal Salient Object Detection                                           | TCSVT  | ResNet34                 | [Paper](https://ieeexplore.ieee.org/document/9801871) / [代码](https://git.openi.org.cn/OpenVision/CCFENet)                  |
| 2022 | ECFFNet        | ECFFNet: Effective and Consistent Feature Fusion Network for RGB-T Salient Object Detection                                          | TSCVT  | ResNet-34                | [Paper](https://ieeexplore.ieee.org/document/9420662) / [代码](https://pan.baidu.com/share/init?surl=Cp6RQMwX3GOTdn3PNyQ72A) |
| 2022 | CSRNet         | Efficient Context-Guided Stacked Refinement Network for RGB-T Salient Object Detection                                               | TCSVT  | ESPNetv2                 | [Paper](https://ieeexplore.ieee.org/document/9505635) / [代码](https://github.com/huofushuo/CSRNet)                          |
| 2022 | SwinNet        | SwinNet: Swin Transformer Drives Edge-Aware RGB-D and RGB-T Salient Object Detection                                                 | TCSVT  | Swin Transformer         | [Paper](https://ieeexplore.ieee.org/document/9611276) / [代码](https://github.com/liuzywen/SwinNet)                          |
| 2022 | APNet          | APNet: Adversarial Learning Assistance and Perceived Importance Fusion Network for All-Day RGB-T Salient Object Detection            | TETIC  | VGG16                    | [Paper](https://ieeexplore.ieee.org/document/9583676)                                                                      |
| 2022 | OSRNet         | Real-Time One-Stream Semantic-Guided Refinement Network for RGB-Thermal Salient Object Detection                                     | TIM    | VGG16 / ResNet-50        | [Paper](https://ieeexplore.ieee.org/document/9803225) / [代码](https://github.com/huofushuo/OSRNet)                          |
| 2022 | DCNet          | Weakly Alignment-Free RGBT Salient Object Detection With Deep Correlation Network                                                    | TIP    | VGG16                    | [Paper](https://ieeexplore.ieee.org/document/9779787)                                                                      |
| 2022 | CFRNet         | RGB-T salient object detection via CNN feature and result saliency map fusion                                                        | AI     | VGG16                    | [Paper](https://link.springer.com/article/10.1007/s10489-021-02984-1)                                                                      |
| 2022 | MIA-DPD        | Multi-modal Interactive Attention and Dual Progressive Decoding Network for RGB-D/T Salient Object Detection                         | NC     | ResNet50                 | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231222002971) / [代码](https://github.com/Liangyh18/MIA_DPD)                         |
| 2022 | -              | Unidirectional RGB-T salient object detection with intertwined driving of encoding and fusion                                        | EAAI   | Segformer                | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197622002743)                                                                      |
| 2022 | SwinMCNet      | Mirror Complementary Transformer Network for RGB-thermal Salient Object Detection                                                    | arXiv  | Swin-B                   | [Paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12221) / [代码](https://github.com/jxr326/SwinMCNet)                          |
| 2022 | EAF-Net        | EAF-Net: an enhancement and aggregation–feedback network for RGB-T salient object detection                                          | MVA    | ResNet-50                | [Paper](https://link.springer.com/article/10.1007/s00138-022-01312-y)                                                                      |
| 2022 | -              | Enabling modality interactions for RGB-T salient object detection                                                                    | CVIU   | ResNet50                 | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314222001047)                                                                      |
| 2022 | MCFNet         | Modal complementary fusion network for RGB-T salient object detection                                                                | AI     | ResNet50                 | [Paper](https://link.springer.com/article/10.1007/s10489-022-03950-1) / [代码](https://github.com/dotaball/MCFNet)                           |
| 2022 | ICANet         | Interactive Context-Aware Network for RGB-T Salient Object Detection                                                                 | arXiv  | ResNet50                 | [Paper](https://arxiv.org/abs/2211.06097)                                                                      |
| 2022 | MFENet         | MFENet: Multitype fusion and enhancement network for detecting salient objects in RGB-T images                                       | DSP    | ResNet34                 | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1051200422004444) / [代码](https://github.com/wujunyi1412/MFENet_DSP)                    |
| 2022 | ACMANet        | Asymmetric cross-modal activation network for RGB-T salient object detection                                                         | KBS    | ResNet50 / ResNet101     | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705122011406) / [代码](https://github.com/xanxuso/ACMANet)                           |
| 2022 | ADFNet         | RGBT Salient Object Detection: A Large-Scale Dataset and Benchmark                                                                   | TMM    | VGG16                    | [Paper](https://ieeexplore.ieee.org/document/9767629) / [代码](https://github.com/lz118/RGBT-Salient-Object-Detection)       |
| 2023 | RGB-T Scribble | Scribble-Supervised RGB-T Salient Object Detection                                                                                   | ICME   | ResNet-50 / PVTv2-B2     | [Paper](https://ieeexplore.ieee.org/document/10219673) / [代码](https://github.com/liuzywen/RGBTScribble-ICME2023)            |
| 2023 | CMDBIF-Net     | Cross-Modality Double Bidirectional Interaction and Fusion Network for RGB-T Salient Object Detection                                | TCSVT  | ResNet50                 | [Paper](https://ieeexplore.ieee.org/document/10032588)                                                                      |
| 2023 | HRTransNet     | HRTransNet: HRFormer-Driven Two-Modality Salient Object Detection                                                                    | TCSVT  | HRFormer / ResNet18      | [Paper](https://ieeexplore.ieee.org/document/9869666) / [代码](https://github.com/liuzywen/HRTransNet)                       |
| 2023 | MITF-Net       | Modality-Induced Transfer-Fusion Network for RGB-D and RGB-T Salient Object Detection                                                | TCSVT  | PVTv2                    | [Paper](https://ieeexplore.ieee.org/document/9925217)                                                                      |
| 2023 | MGAI           | Multiple Graph Affinity Interactive Network and a Variable Illumination Dataset for RGBT Image Salient Object Detection              | TCSVT  | Res2Net50                | [Paper](https://ieeexplore.ieee.org/document/10003255) / [代码](https://github.com/huanglm-me/VI-RGBT1500)                    |
| 2023 | MROS           | Modality Registration and Object Search Framework for UAV-Based Unregistered RGB-T Image Salient Object Detection                    | TGRS   | Res2Net50                | [Paper](https://ieeexplore.ieee.org/document/10315195) / [代码](https://github.com/VDT-2048/UAV-RGB-T-2400)                   |
| 2023 | CAVER          | Cross-Modal View-Mixed Transformer for Bi-Modal Salient Object Detection                                                             | TIP    | ResNet50 / ResNet101     | [Paper](https://ieeexplore.ieee.org/document/10015667) / [代码](https://github.com/lartpang/caver)                            |
| 2023 | PRLNet         | Position-Aware Relation Learning for RGB-Thermal Salient Object Detection                                                            | TIP    | Swin                     | [Paper](https://ieeexplore.ieee.org/document/10113883)                                                                      |
| 2023 | WaveNet        | WaveNet: Wavelet Network With Knowledge Distillation for RGB-T Salient Object Detection                                              | TIP    | Wave-MLP                 | [Paper](https://ieeexplore.ieee.org/document/10127616) / [代码](https://github.com/nowander/WaveNet)                          |
| 2023 | TNet           | Does Thermal Really Always Matter for RGB-T Salient Object Detection?                                                                | TMM    | ResNet50                 | [Paper](https://ieeexplore.ieee.org/document/9926193) / [代码](https://github.com/rmcong/TNet_TMM2022)                       |
| 2023 | LSNet          | LSNet: Lightweight Spatial Boosting Network for Detecting Salient Objects in RGB-Thermal Images                                      | TIP    | MobileNetV2              | [Paper](https://ieeexplore.ieee.org/document/10042233) / [代码](https://github.com/zyrant/LSNet)                              |
| 2023 | EAEF           | Explicit Attention-Enhanced Fusion for RGB-Thermal Perception Tasks                                                                  | RAL    | ResNet50 / ResNet152     | [Paper](https://ieeexplore.ieee.org/document/10113725) / [代码](https://github.com/freeformrobotics/eaefnet)                  |
| 2023 | TAGFNet        | Thermal images-aware guided early fusion network for cross-illumination RGB-T salient object detection                               | EAAI   | VGG16                    | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197622006303) / [代码](https://github.com/VDT-2048/TAGFNet)                          |
| 2023 | FANet          | Feature aggregation with transformer for RGB-T salient object detection                                                              | NC     | VGG19                    | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231223004526) / [代码](https://github.com/ELOESZHANG/FANet)                          |
| 2023 | MENet          | MENet: Lightweight multimodality enhancement network for detecting salient objects in RGB-thermal images                             | NC     | MobileNetV2              | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231223000358)                                                                      |
| 2023 | -              | Cross-modal co-feedback cellular automata for RGB-T saliency detection                                                               | PR     | VGG19                    | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320322006185)                                                                      |
| 2023 | EFNet          | Feature Enhancement and Fusion for RGB-T Salient Object Detection                                                                    | ICIP   | Swin-B                   | [Paper](https://ieeexplore.ieee.org/document/10222404)                                                                      |
| 2023 | AiOSOD         | All in One: RGB, RGB-D, and RGB-T Salient Object Detection                                                                           | arXiv  | T2T-ViT-10               | [Paper](https://arxiv.org/abs/2311.14746)                                                                      |
| 2023 | SPNet          | Saliency Prototype for RGB-D and RGB-T Salient Object Detection                                                                      | ACM MM | PVT                      | [Paper](https://dl.acm.org/doi/10.1145/3581783.3612466) / [代码](https://github.com/2490o/SPNet)                               |
| 2023 | FFANet         | Frequency-aware feature aggregation network with dual-task consistency for RGB-T salient object detection                            | PR     | Swin-B                   | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320323007409)                                                                      |
| 2023 | IRFS           | An interactively reinforced paradigm for joint infrared-visible image fusion and saliency object detection                           | IF     | ResNet-34                | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1566253523001446) / [代码](https://github.com/wdhudiekou/IRFS)                           |
| 2023 | TIDNet         | Three-stream interaction decoder network for RGB-thermal salient object detection                                                    | KBS    | VGG16 / ResNet-50        | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705122011005)                                                                      |
| 2024 | VSCode         | VSCode: General Visual Salient and Camouflaged Object Detection with 2D Prompt Learning                                              | CVPR   | Swin                     | [Paper](https://arxiv.org/pdf/2311.15011) / [代码](https://github.com/Sssssuperior/VSCode)                       |
| 2024 | TCINet         | Transformer-Based Cross-Modal Integration Network for RGB-T Salient Object Detection                                                 | TCE    | Swin                     | [Paper](https://ieeexplore.ieee.org/document/10504918) / [代码](https://github.com/lvchengtao/TCINet)                         |
| 2024 | -              | RGB-T Saliency Detection Based on Multiscale Modal Reasoning Interaction                                                             | TIM    | VGG19                    | [Paper](https://ieeexplore.ieee.org/document/10638515)                                                                      |
| 2024 | UTDNet         | UTDNet: A unified triplet decoder network for multimodal salient object detection                                                    | NN     | VGG16 / ResNet50         | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006755)                                                                      |
| 2024 | MSEDNet        | MSEDNet: Multi-scale fusion and edge-supervised network for RGB-T salient object detection                                           | NN     | ResNet34/50/101/152      | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023007384) / [代码](https://github.com/Zhou-wy/MSEDNet)                           |
| 2024 | SACNet         | Alignment-Free RGBT Salient Object Detection: Semantics-guided Asymmetric Correlation Network and A Unified Benchmark                | TMM    | Swin-B                   | [Paper](https://ieeexplore.ieee.org/document/10551543) / [代码](https://github.com/Angknpng/SACNet)                           |
| 2024 | VST++          | VST++: Efficient and Stronger Visual Saliency Transformer                                                                            | TPAMI  | T2T-ViT                  | [Paper](https://ieeexplore.ieee.org/document/10497889) / [代码](https://github.com/nnizhang/VST)                              |
| 2024 | UniTR          | UniTR: A Unified TRansformer-based Framework for Co-object and Multi-modal Saliency Detection                                        | TMM    | Swin                     | [Paper](https://ieeexplore.ieee.org/document/10444934)                                                                      |
| 2024 | LAFB           | Learning Adaptive Fusion Bank for Multi-Modal Salient Object Detection                                                               | TCSVT  | Res2Net-50               | [Paper](https://ieeexplore.ieee.org/document/10464332)                                                                      |
| 2024 | WGOFNet        | Weighted Guided Optional Fusion Network for RGB-T Salient Object Detection                                                           | ToMM   | PVT                      | [Paper](https://dl.acm.org/doi/full/10.1145/3624984) / [代码](https://github.com/WJ-CV/WGOFNet)                             |
| 2025 | MFAGAN         | Mirror Feature-Aware Generative Adversarial Network for RGB-T Salient Object Detection                                               | ICIP   | ConvNeXt V2              | [Paper](https://ieeexplore.ieee.org/document/11084456) / [代码](https://github.com/asd291614761/MFAGAN)                       |
| 2025 | LGPNet         | Rethinking Lightweight RGB–Thermal Salient Object Detection With Local and Global Perception Network                                 | IoTJ   | MobileViT-XS             | [Paper](https://ieeexplore.ieee.org/document/10877848)                                                                      |
| 2025 | DSCDNet        | A Dual-Stream Cross-Domain Integration Network for RGB-T Salient Object Detection                                                    | TCE    | ConvNeXt-B               | [Paper](https://ieeexplore.ieee.org/document/10758759)                                                                      |
| 2025 | EDEF           | Explicitly Disentangling and Exclusively Fusing for Semi-Supervised Bi-Modal Salient Object Detection                                | TCSVT  | PVT                      | [Paper](https://ieeexplore.ieee.org/document/10788520) / [代码](https://github.com/WJ-CV/EDEFNet-IEEE-TCSVT)                  |
| 2025 | ISMNet         | Intra-Modality Self-Enhancement Mirror Network for RGB-T Salient Object Detection                                                    | TCSVT  | PVT                      | [Paper](https://ieeexplore.ieee.org/document/10740324)                                                                      |
| 2025 | UniSOD         | Unified-modal Salient Object Detection via Adaptive Prompt Learning                                                                  | TCSVT  | Swin-B                   | [Paper](https://ieeexplore.ieee.org/document/11082344)                                                                      |
| 2025 | CCUENet        | Collaborating Constrained and Unconstrained Encodings for Cross-Modal Salient Object Detection                                       | TETCI  | MobileNetV3              | [Paper](https://ieeexplore.ieee.org/document/11038751)                                                                      |
| 2025 | AlignSal       | Efficient Fourier Filtering Network With Contrastive Learning for AAV-Based Unaligned Bimodal Salient Object Detection               | TGRS   | CDFFormer-S18            | [Paper](https://ieeexplore.ieee.org/document/10975009) / [代码](https://github.com/JoshuaLPF/AlignSal)                        |
| 2025 | DFINet         | Cognition-Inspired Dynamic Feature Integration Network for RGB-D and RGB-T Salient Object Detection                                  | TIM    | Swin                     | [Paper](https://ieeexplore.ieee.org/document/11131311)                                                                      |
| 2025 | TCINet         | Three-Decoder Cross-Modal Interaction Network for Unregistered RGB-T Salient Object Detection                                        | TIM    | Swin                     | [Paper](https://ieeexplore.ieee.org/document/10947101) / [代码](https://github.com/zqiuqiu235/TCINet)                         |
| 2025 | CONTRINET      | Divide-and-Conquer: Confluent Triple-Flow Network for RGB-T Salient Object Detection                                                 | TPAMI  | VGG16 / Res2Net50 / Swin | [Paper](https://ieeexplore.ieee.org/document/10778650) / [代码](https://github.com/CSer-Tang-hao/ConTriNet_RGBT-SOD)          |
| 2025 | Samba          | Samba: A Unified Mamba-based Framework for General Salient Object Detection                                                          | CVPR   | VMamba                   | [Paper](https://ieeexplore.ieee.org/abstract/document/11093604) / [代码](https://github.com/Jia-hao999/Samba)                          |
| 2025 | TwinsTNet      | TwinsTNet: Broad-View Twins Transformer Network for Bi-Modal Salient Object Detection                                                | TIP    | Swin-B                   | [Paper](https://ieeexplore.ieee.org/abstract/document/10982382) / [代码](https://github.com/JoshuaLPF/TwinsTNet)                       |
| 2025 | PCNet          | Alignment-Free RGB-T Salient Object Detection: A Large-scale Dataset and Progressive Correlation Network                             | AAAI   | Swin-B                   | [Paper](https://dl.acm.org/doi/10.1609/aaai.v39i7.32838) / [代码](https://github.com/Angknpng/PCNet)                            |
| 2025 | UniRGB-IR      | UniRGB-IR: A Unified Framework for Visible-Infrared Downstream Tasks via Adapter Tuning                                              | ACMMM  | ViT-Base                 | [Paper](https://dl.acm.org/doi/10.1145/3746027.3754806) / [代码](https://github.com/PoTsui99/UniRGB-IR)                        |
| 2025 | HSMNet         | Hierarchical semantics guided multi-scale correlation network for alignment-free red-green-blue and thermal salient object detection | EAAI   | Swin                     | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197625024029)                                                                      |


### Other Paradigms

#### Vision Foundation Model-based Methods

| Year | Method  | Title                                                                                               | Pub. | Backbone              | Resources |
| ---- | ------- | --------------------------------------------------------------------------------------------------- | ----- | --------------------- | --------- |
| 2025 | KAN-SAM | KAN-SAM: Kolmogorov-Arnold Network Guided Segment Anything Model for RGB-T Salient Object Detection | arXiv | SAM2 + KAN adapter    | [Paper](https://arxiv.org/abs/2504.05878) |
| 2025 | HyPSAM  | HyPSAM: Hybrid Prompt-driven Segment Anything Model for RGB-Thermal Salient Object Detection        | TCSVT | Swin Transformer V2-B | [Paper](https://ieeexplore.ieee.org/document/11177578) |


#### Diffusion Model-based Methods

| Year | Method  | Title                                                                        | Pub.  | Backbone                            | Resources                                                     |
| ---- | ------- | ---------------------------------------------------------------------------- | ------ | ----------------------------------- | ------------------------------------------------------------- |
| 2025 | diffSOD | Multi-modal Salient Object Detection via a Unified Diffusion Model           | ICASSP | Diffusion & De-ViT (Deformable ViT) | [Paper](https://ieeexplore.ieee.org/document/10887966) / [代码](https://github.com/QuantumScriptHub/diffSOD) |
| 2025 | DiMSOD  | DiMSOD: A Diffusion-Based Framework for Multi-Modal Salient Object Detection | AAAI   | Stable Diffusion + PVT              | [Paper](https://dl.acm.org/doi/abs/10.1609/aaai.v39i10.33096)                                                     |


---

## 🗂 Datasets

| Dataset | Alignment | Year | Pub. | Size | #Obj. | Types | Sensor | Resolution | Resources |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VT821** | Aligned | 2018 | TCSVT | 821 | Variable | Indoor/outdoor | MAG32 & SONY TD-2073 | 480 × 640 | [Link](https://github.com/lz118/RGBT-Salient-Object-Detection) |
| **VT1000** | Aligned | 2019 | TMM | 1,000 | Variable | Indoor/outdoor | FLIR SC620 | 480 × 640 | [Link](https://github.com/lz118/RGBT-Salient-Object-Detection) |
| **VT5000** | Aligned | 2022 | TMM | 5,000 | Variable | Indoor/outdoor | FLIR T640 & T610 | 640 × 480 | [Link](https://github.com/lz118/RGBT-Salient-Object-Detection) |
| **UAV RGB-T 2400** | Unaligned | 2023 | TGRS | 2,400 | Variable | Outdoor (UAV-view) | DJI MAVIC 2 Enterprise Advanced | 1920×1080 (RGB) & 640×512 (T) | [Link](https://github.com/VDT-2048/UAV-RGB-T-2400) |
| **UVT20K** | Unaligned | 2025 | AAAI | 20,000 | Variable | Indoor & Outdoor (real-world scenes) | Hikvision DS-2TP23 & FLIR SC620 | Various | [Link](https://github.com/Angknpng/PCNet) |


---

## 📐 Evaluation Metrics

Commonly used evaluation metrics in RGB-T SOD include:

* **Precision–Recall (PR) curves**
* **F-measure ($F_{\beta}$)**
* **Weighted F-measure ($F_{\beta}^{\omega}$)**
* **Mean Absolute Error (MAE)**
* **Structure-measure ($S_{\alpha}$)**
* **Enhanced-alignment measure ($E_{\phi}$)**

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
  year   = {2025}
}
```
