# Awesome RGB-T Salient Object Detection

A curated list of **RGB-Thermal Salient Object Detection (RGB-T SOD)** methods, datasets, and related resources.

RGB-T SOD leverages the complementary information between visible (RGB) and thermal infrared modalities to achieve robust salient object detection under challenging conditions such as low illumination, camouflage, occlusion, and background clutter.

---

## Contents

- [Papers](#papers)
  - [RGB-T Salient Object Detection](#rgb-t-salient-object-detection)
    - [Machine Learning-based Methods](#machine-learning-based-methods)
    - [Deep Learning-based Methods](#deep-learning-based-methods)
    - [Other Methods](#other-methods)
- [Datasets](#datasets)
- [Evaluation Metrics](#evaluation-metrics)
- [Reference](#reference)

---

## Papers

### RGB-T Salient Object Detection

#### Machine Learning-based Methods

| Year | Method | Title | Pub |代码|
|------|--------|-------|-------|----------|------|
| 2018 | MTMR | RGB-T saliency detection benchmark: Dataset, baselines, analysis and a novel approach | IGTA | [Link](https://github.com/lz118/RGBT-Salient-Object-Detection) |
| 2019 | M3S-NIR | M3S-NIR: Multi-Modal Multi-Scale Noise-Insensitive Ranking for RGB-T Saliency Detection | MIPR | [Link](https://github.com/lz118/RGBT-Salient-Object-Detection/tree/master/Code/M3S-NIR) |
| 2019 | SGDL | RGB-T Image Saliency Detection via Collaborative Graph Learning | TMM | [Link](https://github.com/lz118/RGBT-Salient-Object-Detection) |
| 2020 | LTCR | RGB-T Saliency Detection via Low-Rank Tensor Learning and Unified Collaborative Ranking | SPL | [Link](https://github.com/huanglm-me/LTCR) |
| 2022 | MGFL | Multi-Graph Fusion and Learning for RGBT Image Saliency Detection | TCSVT | [Link](https://github.com/lmhuang-me/RGBT_MGFL) |

---

#### Deep Learning-based Methods

| Year | Method | Title | Pub | Backbone |Resources|
|------|--------|-------|-------|----------|------|
| 2017 | - | Learning Multiscale Deep Features and SVM Regressors for Adaptive RGB-T Saliency Detection | ISCID | VGG16 |  |
| 2020 | FMCF | RGB-T Salient Object Detection via Fusing Multi-Level CNN Features | TIP | VGG16 |  |
| 2020 | - | Multi-Spectral Salient Object Detection by Adversarial Domain Adaptation  | AAAI  | VGG16 | [Link](https://tsllb.github.io/MultiSOD.html) |
| 2020 | - | Deep Domain Adaptation Based Multi-spectral Salient Object Detection | TMM | VGG16 | [Link](https://tsllb.github.io/MultiSOD.html) |
| 2021 | TSFNet | TSFNet: Two-Stage Fusion Network for RGB-T Salient Object Detection | SPL | ResNet-34  |  |
| 2021 | FFNet | Revisiting Feature Fusion for RGB-T Salient Object Detection | TCSVT | VGG16 |  |
| 2021 | MIDD | Multi-Interactive Dual-Decoder for RGB-Thermal Salient Object Detection | TIP | VGG16 | [Link](https://github.com/lz118/Multi-interactive-Dual-decoder) |
| 2021 | MMNet | Unified Information Fusion Network for Multi-Modal RGB-D and RGB-T Salient Object Detection | TCSVT | VGG19 / Res2Net-50 |  |
| 2021 | - | Salient Target Detection in RGB-T Image based on Multi-level Semantic Information | CYBER | VGG16 |  |
| 2022 | CGFNet | CGFNet: Cross-Guided Fusion Network for RGB-T Salient Object Detection | TCSVT | VGG16 | [Link](https://github.com/wangjie0825/CGFNet.git) |
| 2022 | CGMDRNet | CGMDRNet: Cross-Guided Modality Difference Reduction Network for RGB-T Salient Object Detection | TCSVT | Res2Net-50 |  |
| 2022 | CCFENet | Cross-Collaborative Fusion-Encoder Network for Robust RGB-Thermal Salient Object Detection | TCSVT | ResNet34  | [Link](https://git.openi.org.cn/OpenVision/CCFENet) |
| 2022 | ECFFNet | ECFFNet: Effective and Consistent Feature Fusion Network for RGB-T Salient Object Detection | TSCVT | ResNet-34  | [Link](https://pan.baidu.com/share/init?surl=Cp6RQMwX3GOTdn3PNyQ72A (提取码: tx48)) |
| 2022 | CSRNet | Efficient Context-Guided Stacked Refinement Network for RGB-T Salient Object Detection | TCSVT | ESPNetv2 | [Link](https://github.com/huofushuo/CSRNet) |
| 2022 | SwinNet | SwinNet: Swin Transformer Drives Edge-Aware RGB-D and RGB-T Salient Object Detection | TCSVT | swin Transformer | [Link](https://github.com/liuzywen/SwinNet) |
| 2022 | APNet | APNet: Adversarial Learning Assistance and Perceived Importance Fusion Network for All-Day RGB-T Salient Object Detection | TETIC | VGG16 |  |
| 2022 | OSRNet | Real-Time One-Stream Semantic-Guided Refinement Network for RGB-Thermal Salient Object Detection（C. Attention Mechanism） | TIM |  VGG16 /ResNet-50 | [Link](https://github.com/huofushuo/OSRNet) |
| 2022 | DCNet | Weakly Alignment-Free RGBT Salient Object Detection With Deep Correlation Network | TIP | VGG16 |  |
| 2022 | CFRNet | RGB-T salient object detection via CNN feature and result saliency map fusion | AI | VGG16 |  |
| 2022 | MIA-DPD | Multi-modal Interactive Attention and Dual Progressive Decoding Network for RGB-D/T Salient Object Detection | NC | ResNet50 | [Link](https://github.com/Liangyh18/MIA_DPD) |
| 2022 | - | Unidirectional RGB-T salient object detection with intertwined driving of encoding and fusion | EAAI | Segformer |  |
| 2022 | SwinMCNet | Mirror Complementary Transformer Network for RGB-thermal Salient Object Detection | arXiv | Swin-B | [Link](https://github.com/jxr326/SwinMCNet) |
| 2022 | EAF-Net | EAF-Net: an enhancement and aggregation–feedback network for RGB-T salient object detection | MVA (4区) | ResNet-50 |  |
| 2022 | - | Enabling modality interactions for RGB-T salient object detection | CVIU | ResNet50 |  |
| 2022 | MCFNet | Modal complementary fusion network for RGB-T salient object detection | AI | ResNet50 | [Link](https://github.com/dotaball/MCFNet) |
| 2022 | ICANet | Interactive Context-Aware Network for RGB-T Salient Object Detection | arXiv | ResNet50 |  |
| 2022 | MFENet | MFENet: Multitype fusion and enhancement network for detecting salient objects in RGB-T images | DSP | ResNet34 | [Link](https://github.com/wujunyi1412/MFENet_DSP) |
| 2022 | ACMANet | Asymmetric cross-modal activation network for RGB-T salient object detection | KBS | ResNet50 / ResNet101 | [Link](https://github.com/xanxuso/ACMANet) |
| 2022 | ADFNet | RGBT Salient Object Detection: A Large-Scale Dataset and Benchmark | TMM | VGG16 | [Link](https://github.com/lz118/RGBT-Salient-Object-Detection) |
| 2023 | RGB-T Scribble | Scribble-Supervised RGB-T Salient Object Detection | ICME | ResNet-50 和 PVTv2-B2 | [Link](https://github.com/liuzywen/RGBTScribble-ICME2023) |
| 2023 | CMDBIF-Net | Cross-Modality Double Bidirectional Interaction and Fusion Network for RGB-T Salient Object Detection | TCSVT | ResNet50 |  |
| 2023 | HRTransNet | HRTransNet: HRFormer-Driven Two-Modality Salient Object Detection | TCSVT | HRFormer&ResNet18 | [Link](https://github.com/liuzywen/HRTransNet) |
| 2023 | MITF-Net | Modality-Induced Transfer-Fusion Network for RGB-D and RGB-T Salient Object Detection | TCSVT | PVTv2 |  |
| 2023 | MGAI | Multiple Graph Affinity Interactive Network and a Variable Illumination Dataset for RGBT Image Salient Object Detection | TCSVT | Res2Net50 | [Link](https://github.com/huanglm-me/VI-RGBT1500) |
| 2023 | MROS | Modality Registration and Object Search Framework for UAV-Based Unregistered RGB-T Image Salient Object Detection | TGRS | Res2Net50 | [Link](https://github.com/VDT-2048/UAV-RGB-T-2400) |
| 2023 | caver | CAVER: Cross-Modal View-Mixed Transformer for Bi-Modal Salient Object Detection | TIP | ResNet50 / ResNet101 | [Link](https://github.com/lartpang/caver) |
| 2023 | PRLNet | Position-Aware Relation Learning for RGB-Thermal Salient Object Detection | TIP | Swin |  |
| 2023 | WaveNet | WaveNet: Wavelet Network With Knowledge Distillation for RGB-T Salient Object Detection | TIP | Wave-MLP | [Link](https://github.com/nowander/WaveNet) |
| 2023 | TNet | Does Thermal Really Always Matter for RGB-T Salient Object Detection? | TMM | ResNet50 | [Link](https://github.com/rmcong/TNet_TMM2022) |
| 2023 | LSNet | LSNet: Lightweight Spatial Boosting Network for Detecting Salient Objects in RGB-Thermal Images | TIP | MobileNetV2 | [Link](https://github.com/zyrant/LSNet) |
| 2023 | EAEF | Explicit Attention-Enhanced Fusion for RGB-Thermal Perception Tasks | RAL | ResNet50 / ResNet152 | [Link](https://github.com/freeformrobotics/eaefnet) |
| 2023 | TAGFNet | Thermal images-aware guided early fusion network for cross-illumination RGB-T salient object detection | EAAI | VGG16 | [Link](https://github.com/VDT-2048/TAGFNet) |
| 2023 | FANet | Feature aggregation with transformer for RGB-T salient object detection | NC | VGG19  | [Link](https://github.com/ELOESZHANG/FANet) |
| 2023 | MENet | MENet: Lightweight multimodality enhancement network for detecting salient objects in RGB-thermal images | NC | MobileNetV2  |  |
| 2023 | - | Cross-modal co-feedback cellular automata for RGB-T saliency detection | PR | VGG19  |  |
| 2023 | EFNet | Feature Enhancement and Fusion for RGB-T Salient Object Detection | ICIP | Swin-B |  |
| 2023 | AiOSOD | All in One: RGB, RGB-D, and RGB-T Salient Object Detection | arXiv | T2T-ViT-10 |  |
| 2023 | SPNet | Saliency Prototype for RGB-D and RGB-T Salient Object Detection | ACM MM | ＰＶＴ | [Link](https://github.com/2490o/SPNet) |
| 2023 | FFANet | Frequency-aware feature aggregation network with dual-task consistency for RGB-T salient object detection | PR | Swin-B |  |
| 2023 | IRFS | An interactively reinforced paradigm for joint infrared-visible image fusion and saliency object detection | IF | ResNet-34 | [Link](https://github.com/wdhudiekou/IRFS) |
| 2023 | TIDNet | Three-stream interaction decoder network for RGB-thermal salient object detection | KBS |  VGG16 或 ResNet-50 |  |
| 2024 | VSCode | VSCode: General Visual Salient and Camouflaged Object Detection with 2D Prompt Learning | CVPR | Swin | [Link](https://github.com/Sssssuperior/VSCode) |
| 2024 | TCINet | Transformer-Based Cross-Modal Integration Network for RGB-T Salient Object Detection | TCE | Swin | [Link](https://github.com/lvchengtao/TCINet) |
| 2024 | - | RGB-T Saliency Detection Based on Multiscale Modal Reasoning Interaction | TIM | VGG19 |  |
| 2024 | UTDNet | UTDNet: A unified triplet decoder network for multimodal salient object detection | NN | VGG-16 / ResNet-50 |  |
| 2024 | MSEDNet | MSEDNet: Multi-scale fusion and edge-supervised network for RGB-T salient object detection | NN | ResNet34/50/101/152 | [Link](https://github.com/Zhou-wy/MSEDNet) |
| 2024 | SACNet | Alignment-Free RGBT Salient Object Detection: Semantics-guided Asymmetric Correlation Network and A Unified Benchmark | TMM | Swin-B | [Link](https://github.com/Angknpng/SACNet) |
| 2024 | VST++ | VST++: Efficient and Stronger Visual Saliency Transformer | TPAMI |  T2T-ViT | [Link](https://github.com/nnizhang/VST) |
| 2024 | UniTR | UniTR: A Unified TRansformer-based Framework for Co-object and Multi-modal Saliency Detection | TMM |  Swin |  |
| 2024 | LAFB | Learning Adaptive Fusion Bank for Multi-Modal Salient Object Detection | TCSVT | Res2Net-50 |  |
| 2024 | WGOFNet | Weighted Guided Optional Fusion Network for RGB-T Salient Object Detection | ToMM | PVT | [Link](https://github.com/WJ-CV/WGOFNet) |
| 2025 | MFAGAN | Mirror Feature-Aware Generative Adversarial Network for RGB-T Salient Object Detection | ICIP | ConvNeXt V2 | [Link](https://github.com/asd291614761/MFAGAN) |
| 2025 | LGPNet | Rethinking Lightweight RGB–Thermal Salient Object Detection With Local and Global Perception Network | IoTJ | MobileViT-XS |  |
| 2025 | DSCDNet | A Dual-Stream Cross-Domain Integration Network for RGB-T Salient Object Detection | TCE | ConvNeXt-B |  |
| 2025 | EDEF | Explicitly Disentangling and Exclusively Fusing for Semi-Supervised Bi-Modal Salient Object Detection | TCSVT | PVT | [Link](https://github.com/WJ-CV/EDEFNet-IEEE-TCSVT) |
| 2025 | ISMNet | Intra-Modality Self-Enhancement Mirror Network for RGB-T Salient Object Detection | TCSVT | PVT |  |
| 2025 | UniSOD | Unified-modal Salient Object Detection via Adaptive Prompt Learning | TCSVT | Swin-B |  |
| 2025 | CCUENet | Collaborating Constrained and Unconstrained Encodings for Cross-Modal Salient Object Detection | TETCI | MobileNetV3 |  |
| 2025 | AlignSal | Efficient Fourier Filtering Network With Contrastive Learning for AAV-Based Unaligned Bimodal Salient Object Detection | TGRS | CDFFormer-S18 | [Link](https://github.com/JoshuaLPF/AlignSal) |
| 2025 | DFINet | Cognition-Inspired Dynamic Feature Integration Network for RGB-D and RGB-T Salient Object Detection | TIM | Swin |  |
| 2025 | TCINet | Three-Decoder Cross-Modal Interaction Network for Unregistered RGB-T Salient Object Detection | TIM | Swin | [Link](https://github.com/zqiuqiu235/TCINet) |
| 2025 | CONTRINET | Divide-and-Conquer: Confluent Triple-Flow Network for RGB-T Salient Object Detection | TPAMI | VGG16 / Res2Net50 / Swin | [Link](https://github.com/CSer-Tang-hao/ConTriNet_RGBT-SOD) |
| 2025 | Samba | Samba: A Unified Mamba-based Framework for General Salient Object Detection | CVPR | VMamba | [Link](https://github.com/Jia-hao999/Samba) |
| 2025 | TwinsTNet | TwinsTNet: Broad-View Twins Transformer Network for Bi-Modal Salient Object Detection | TIP | Swin Transformer Base | [Link](https://github.com/JoshuaLPF/TwinsTNet) |
| 2025 | PCNet | Alignment-Free RGB-T Salient Object Detection: A Large-scale Dataset and Progressive Correlation Network | AAAI | Swin Transformer Base／S-Adapter（语义适配器）对 IHN 进行微调 | [Link](https://github.com/Angknpng/PCNet) |
| 2025 | UniRGB-IR. | UniRGB-IR: A Unified Framework for Visible-Infrared Downstream Tasks via Adapter Tuning | ACMMM | ViT-Base | [Link](https://github.com/PoTsui99/UniRGB-IR) |
| 2025 | HSMNet | Hierarchical semantics guided multi-scale correlation network for alignment-free red-green-blue and thermal salient object detection | EAAI |  Swin Transformer  |  |


---

#### Other Methods

##### Vision Foundation Model-based Methods

| Year | Method | Title | Pub | Backbone |Resources|
|------|--------|-------|-------|----------|------|
| 2025 | KAN-SAM | KAN-SAM: Kolmogorov-Arnold Network Guided Segment Anything Model for RGB-T Salient Object Detection | arXiv | SAM2+KAN adapter |  |
| 2025 | HyPSAM | HyPSAM:Hybrid Prompt-driven Segment Anything Model for RGB-Thermal Salient Object Detection | TCSVT | Swin Transformer V2-B |  |

##### Diffusion Model-based Methods

| Year | Method | Title | Pub | Backbone |Resources|
|------|--------|-------|-------|----------|------|
| 2025 | diffSOD | Multi-modal Salient Object Detection via a Unified Diffusion Model | ICASSP | Diffusion & De-ViT（Deformable ViT）  | [Link](https://github.com/QuantumScriptHub/diffSOD) |
| 2025 | DiMSOD | DiMSOD: A Diffusion-Based Framework for Multi-Modal Salient Object Detection | AAAI | Stable Diffusion + PVT |  |

---

## Datasets

| Dataset | Modalities | Size | Scene |
|--------|------------|------|-------|
| VT5000 | RGB-T | 5000 | Indoor / Outdoor |
| VT1000 | RGB-T | 1000 | Outdoor |


---

## Related Surveys Recommended


