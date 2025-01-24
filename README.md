# awesome-feed-forward-reconstruction

A curated list of papers and open-source resources focused on reconstruction based on large feed-forward model. 

### 1. [NIPS '2024] SCube: Instant Large-Scale Scene Reconstruction using VoxSplats
**Authors**: Xuanchi Ren, Yifan Lu, Hanxue Liang, Zhangjie Wu, Huan Ling, Mike Chen, Sanja Fidler, Francis Williams, Jiahui Huang
<details span>
<summary><b>Abstract</b></summary>
  We present SCube, a novel method for reconstructing large-scale 3D scenes (geometry, appearance, and semantics) from a sparse set of posed images. Our method encodes reconstructed scenes using a novel representation VoxSplat, which is a set of 3D Gaussians supported on a high-resolution sparse-voxel scaffold. To reconstruct a VoxSplat from images, we employ a hierarchical voxel latent diffusion model conditioned on the input images followed by a feedforward appearance prediction model. The diffusion model generates high-resolution grids progressively in a coarse-to-fine manner, and the appearance network predicts a set of Gaussians within each voxel. From as few as 3 non-overlapping input images, SCube can generate millions of Gaussians with a 1024^3 voxel grid spanning hundreds of meters in 20 seconds. Past works tackling scene reconstruction from images either rely on per-scene optimization and fail to reconstruct the scene away from input views (thus requiring dense view coverage as input) or leverage geometric priors based on low-resolution models, which produce blurry results. In contrast, SCube leverages high-resolution sparse networks and produces sharp outputs from few views. We show the superiority of SCube compared to prior art using the Waymo self-driving dataset on 3D reconstruction and demonstrate its applications, such as LiDAR simulation and text-to-scene generation.
</details>

  [📄 Paper](https://arxiv.org/pdf/2410.20030) | [💻 Code](https://github.com/nv-tlabs/SCube) | [🌐 Project Page](https://research.nvidia.com/labs/toronto-ai/scube/)  


### 2. [NIPS '2024] Large Spatial Model: End-to-end Unposed Images to Semantic 3D
**Authors**: Zhiwen Fan, Jian Zhang, Wenyan Cong, Peihao Wang, Renjie Li, Kairun Wen, Shijie Zhou, Achuta Kadambi, Zhangyang Wang, Danfei Xu, Boris Ivanovic, Marco Pavone, Yue Wang
<details span>
<summary><b>Abstract</b></summary>
  Reconstructing and understanding 3D structures from a limited number of images is a well-established problem in computer vision. Traditional methods usually break this task into multiple subtasks, each requiring complex transformations between different data representations. For instance, dense reconstruction through Structure-from-Motion (SfM) involves converting images into key points, optimizing camera parameters, and estimating structures. Afterward, accurate sparse reconstructions are required for further dense modeling, which is subsequently fed into task-specific neural networks. This multi-step process results in considerable processing time and increased engineering complexity.
In this work, we present the Large Spatial Model (LSM), which processes unposed RGB images directly into semantic radiance fields. LSM simultaneously estimates geometry, appearance, and semantics in a single feed-forward operation, and it can generate versatile label maps by interacting with language at novel viewpoints. Leveraging a Transformer-based architecture, LSM integrates global geometry through pixel-aligned point maps. To enhance spatial attribute regression, we incorporate local context aggregation with multi-scale fusion, improving the accuracy of fine local details. To tackle the scarcity of labeled 3D semantic data and enable natural language-driven scene manipulation, we incorporate a pre-trained 2D language-based segmentation model into a 3D-consistent semantic feature field. An efficient decoder then parameterizes a set of semantic anisotropic Gaussians, facilitating supervised end-to-end learning. Extensive experiments across various tasks show that LSM unifies multiple 3D vision tasks directly from unposed images, achieving real-time semantic 3D reconstruction for the first time.
</details>

  [📄 Paper](https://arxiv.org/pdf/2410.18956) | 💻 Code | [🌐 Project Page](https://largespatialmodel.github.io/)  

### 3. [CVPR '2024] DUSt3R: Geometric 3D Vision Made Easy
**Authors**: Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud
<details span>
<summary><b>Abstract</b></summary>
  Multi-view stereo reconstruction (MVS) in the wild requires to first estimate the camera parameters e.g. intrinsic and extrinsic parameters. These are usually tedious and cumbersome to obtain, yet they are mandatory to triangulate corresponding pixels in 3D space, which is the core of all best performing MVS algorithms. In this work, we take an opposite stance and introduce DUSt3R, a radically novel paradigm for Dense and Unconstrained Stereo 3D Reconstruction of arbitrary image collections, i.e. operating without prior information about camera calibration nor viewpoint poses. We cast the pairwise reconstruction problem as a regression of pointmaps, relaxing the hard constraints of usual projective camera models. We show that this formulation smoothly unifies the monocular and binocular reconstruction cases. In the case where more than two images are provided, we further propose a simple yet effective global alignment strategy that expresses all pairwise pointmaps in a common reference frame. We base our network architecture on standard Transformer encoders and decoders, allowing us to leverage powerful pretrained models. Our formulation directly provides a 3D model of the scene as well as depth information, but interestingly, we can seamlessly recover from it, pixel matches, relative and absolute camera. Exhaustive experiments on all these tasks showcase that the proposed DUSt3R can unify various 3D vision tasks and set new SoTAs on monocular/multi-view depth estimation as well as relative pose estimation. In summary, DUSt3R makes many geometric 3D vision tasks easy.
</details>

  [📄 Paper](https://arxiv.org/pdf/2312.14132) | [💻 Code](https://github.com/naver/dust3r) | [🌐 Project Page](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/)  


### 4. [Arxiv '2024] Grounding Image Matching in 3D with MASt3R
**Authors**: Vincent Leroy, Yohann Cabon, Jérôme Revaud
<details span>
<summary><b>Abstract</b></summary>
  Image Matching is a core component of all best-performing algorithms and pipelines in 3D vision. Yet despite matching being fundamentally a 3D problem, intrinsically linked to camera pose and scene geometry, it is typically treated as a 2D problem. This makes sense as the goal of matching is to establish correspondences between 2D pixel fields, but also seems like a potentially hazardous choice. In this work, we take a different stance and propose to cast matching as a 3D task with DUSt3R, a recent and powerful 3D reconstruction framework based on Transformers. Based on pointmaps regression, this method displayed impressive robustness in matching views with extreme viewpoint changes, yet with limited accuracy. We aim here to improve the matching capabilities of such an approach while preserving its robustness. We thus propose to augment the DUSt3R network with a new head that outputs dense local features, trained with an additional matching loss. We further address the issue of quadratic complexity of dense matching, which becomes prohibitively slow for downstream applications if not carefully treated. We introduce a fast reciprocal matching scheme that not only accelerates matching by orders of magnitude, but also comes with theoretical guarantees and, lastly, yields improved results. Extensive experiments show that our approach, coined MASt3R, significantly outperforms the state of the art on multiple matching tasks. In particular, it beats the best published methods by 30% (absolute improvement) in VCRE AUC on the extremely challenging Map-free localization dataset.
</details>

  [📄 Paper](https://arxiv.org/pdf/2406.09756) | [💻 Code](https://github.com/naver/mast3r) | [🌐 Project Page](https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/)  


### 5. [Arxiv '2024] MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion
**Authors**: Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, Ming-Hsuan Yang
<details span>
<summary><b>Abstract</b></summary>
  Estimating geometry from dynamic scenes, where objects move and deform over time, remains a core challenge in computer vision. Current approaches often rely on multi-stage pipelines or global optimizations that decompose the problem into subtasks, like depth and flow, leading to complex systems prone to errors. In this paper, we present Motion DUSt3R (MonST3R), a novel geometry-first approach that directly estimates per-timestep geometry from dynamic scenes. Our key insight is that by simply estimating a pointmap for each timestep, we can effectively adapt DUST3R's representation, previously only used for static scenes, to dynamic scenes. However, this approach presents a significant challenge: the scarcity of suitable training data, namely dynamic, posed videos with depth labels. Despite this, we show that by posing the problem as a fine-tuning task, identifying several suitable datasets, and strategically training the model on this limited data, we can surprisingly enable the model to handle dynamics, even without an explicit motion representation. Based on this, we introduce new optimizations for several downstream video-specific tasks and demonstrate strong performance on video depth and camera pose estimation, outperforming prior work in terms of robustness and efficiency. Moreover, MonST3R shows promising results for primarily feed-forward 4D reconstruction.
</details>

  [📄 Paper](https://arxiv.org/pdf/2410.03825) | [💻 Code](https://github.com/Junyi42/monst3r) | [🌐 Project Page](https://monst3r-project.github.io/)  


### 6. [CVPR '2024 Oral, Best Paper Runner-Up] pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction
**Authors**: David Charatan, Sizhe Li, Andrea Tagliasacchi, Vincent Sitzmann
<details span>
<summary><b>Abstract</b></summary>
  We introduce pixelSplat, a feed-forward model that learns to reconstruct 3D radiance fields parameterized by 3D Gaussian primitives from pairs of images. Our model features real-time and memory-efficient rendering for scalable training as well as fast 3D reconstruction at inference time. To overcome local minima inherent to sparse and locally supported representations, we predict a dense probability distribution over 3D and sample Gaussian means from that probability distribution. We make this sampling operation differentiable via a reparameterization trick, allowing us to back-propagate gradients through the Gaussian splatting representation. We benchmark our method on wide-baseline novel view synthesis on the real-world RealEstate10k and ACID datasets, where we outperform state-of-the-art light field transformers and accelerate rendering by 2.5 orders of magnitude while reconstructing an interpretable and editable 3D radiance field.
</details>

  [📄 Paper](https://arxiv.org/pdf/2312.12337) | [💻 Code](https://github.com/dcharatan/pixelsplat) | [🌐 Project Page](https://davidcharatan.com/pixelsplat/)  

### 7. [ECCV '2024 Oral] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images
**Authors**: Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, Jianfei Cai
<details span>
<summary><b>Abstract</b></summary>
  We introduce MVSplat, an efficient model that, given sparse multi-view images as input, predicts clean feed-forward 3D Gaussians. To accurately localize the Gaussian centers, we build a cost volume representation via plane sweeping, where the cross-view feature similarities stored in the cost volume can provide valuable geometry cues to the estimation of depth. We also learn other Gaussian primitives' parameters jointly with the Gaussian centers while only relying on photometric supervision. We demonstrate the importance of the cost volume representation in learning feed-forward Gaussians via extensive experimental evaluations. On the large-scale RealEstate10K and ACID benchmarks, MVSplat achieves state-of-the-art performance with the fastest feed-forward inference speed (22~fps). More impressively, compared to the latest state-of-the-art method pixelSplat, MVSplat uses 10× fewer parameters and infers more than 2× faster while providing higher appearance and geometry quality as well as better cross-dataset generalization.
</details>

  [📄 Paper](https://arxiv.org/pdf/2403.14627) | [💻 Code](https://github.com/donydchen/mvsplat) | [🌐 Project Page](https://donydchen.github.io/mvsplat/)  


### 8. [Arxiv '2024] DepthSplat: Connecting Gaussian Splatting and Depth
**Authors**: Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, Marc Pollefeys
<details span>
<summary><b>Abstract</b></summary>
  Gaussian splatting and single/multi-view depth estimation are typically studied in isolation. In this paper, we present DepthSplat to connect Gaussian splatting and depth estimation and study their interactions. More specifically, we first contribute a robust multi-view depth model by leveraging pre-trained monocular depth features, leading to high-quality feed-forward 3D Gaussian splatting reconstructions. We also show that Gaussian splatting can serve as an unsupervised pre-training objective for learning powerful depth models from large-scale unlabelled datasets. We validate the synergy between Gaussian splatting and depth estimation through extensive ablation and cross-task transfer experiments. Our DepthSplat achieves state-of-the-art performance on ScanNet, RealEstate10K and DL3DV datasets in terms of both depth estimation and novel view synthesis, demonstrating the mutual benefits of connecting both tasks.
</details>

  [📄 Paper](https://arxiv.org/pdf/2410.13862) | [💻 Code](https://github.com/cvg/depthsplat) | [🌐 Project Page](https://haofeixu.github.io/depthsplat/)  


### 9. [CVPR '2021] pixelNeRF: Neural Radiance Fields from One or Few Images
**Authors**: Alex Yu, Vickie Ye, Matthew Tancik, Angjoo Kanazawa
<details span>
<summary><b>Abstract</b></summary>
  We propose pixelNeRF, a learning framework that predicts a continuous neural scene representation conditioned on one or few input images. The existing approach for constructing neural radiance fields involves optimizing the representation to every scene independently, requiring many calibrated views and significant compute time. We take a step towards resolving these shortcomings by introducing an architecture that conditions a NeRF on image inputs in a fully convolutional manner. This allows the network to be trained across multiple scenes to learn a scene prior, enabling it to perform novel view synthesis in a feed-forward manner from a sparse set of views (as few as one). Leveraging the volume rendering approach of NeRF, our model can be trained directly from images with no explicit 3D supervision. We conduct extensive experiments on ShapeNet benchmarks for single image novel view synthesis tasks with held-out objects as well as entire unseen categories. We further demonstrate the flexibility of pixelNeRF by demonstrating it on multi-object ShapeNet scenes and real scenes from the DTU dataset. In all cases, pixelNeRF outperforms current state-of-the-art baselines for novel view synthesis and single image 3D reconstruction.
</details>

  [📄 Paper](https://arxiv.org/pdf/2012.02190) | [💻 Code](https://github.com/sxyu/pixel-nerf) | [🌐 Project Page](https://alexyu.net/pixelnerf/)  


### 10. [ICLR '2023] Is Attention All That NeRF Needs?
**Authors**: Mukund Varma T, Peihao Wang, Xuxi Chen, Tianlong Chen, Subhashini Venugopalan, Zhangyang Wang
<details span>
<summary><b>Abstract</b></summary>
  We present Generalizable NeRF Transformer (GNT), a transformer-based architecture that reconstructs Neural Radiance Fields (NeRFs) and learns to renders novel views on the fly from source views. While prior works on NeRFs optimize a scene representation by inverting a handcrafted rendering equation, GNT achieves neural representation and rendering that generalizes across scenes using transformers at two stages. (1) The view transformer leverages multi-view geometry as an inductive bias for attention-based scene representation, and predicts coordinate-aligned features by aggregating information from epipolar lines on the neighboring views. (2) The ray transformer renders novel views using attention to decode the features from the view transformer along the sampled points during ray marching. Our experiments demonstrate that when optimized on a single scene, GNT can successfully reconstruct NeRF without an explicit rendering formula due to the learned ray renderer. When trained on multiple scenes, GNT consistently achieves state-of-the-art performance when transferring to unseen scenes and outperform all other methods by ~10% on average. Our analysis of the learned attention maps to infer depth and occlusion indicate that attention enables learning a physically-grounded rendering. Our results show the promise of transformers as a universal modeling tool for graphics.
</details>

  [📄 Paper](https://arxiv.org/pdf/2207.13298) | [💻 Code](https://github.com/VITA-Group/GNT) | [🌐 Project Page](https://vita-group.github.io/GNT/)  


### 11. [CVPR '2021] IBRNet: Learning Multi-View Image-Based Rendering
**Authors**: Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul Srinivasan, Howard Zhou, Jonathan T. Barron, Ricardo Martin-Brualla, Noah Snavely, Thomas Funkhouser
<details span>
<summary><b>Abstract</b></summary>
  We present a method that synthesizes novel views of complex scenes by interpolating a sparse set of nearby views. The core of our method is a network architecture that includes a multilayer perceptron and a ray transformer that estimates radiance and volume density at continuous 5D locations (3D spatial locations and 2D viewing directions), drawing appearance information on the fly from multiple source views. By drawing on source views at render time, our method hearkens back to classic work on image-based rendering (IBR), and allows us to render high-resolution imagery. Unlike neural scene representation work that optimizes per-scene functions for rendering, we learn a generic view interpolation function that generalizes to novel scenes. We render images using classic volume rendering, which is fully differentiable and allows us to train using only multi-view posed images as supervision. Experiments show that our method outperforms recent novel view synthesis methods that also seek to generalize to novel scenes. Further, if fine-tuned on each scene, our method is competitive with state-of-the-art single-scene neural rendering methods.
</details>

  [📄 Paper](https://arxiv.org/pdf/2102.13090) | [💻 Code](https://github.com/googleinterns/IBRNet) | [🌐 Project Page](https://ibrnet.github.io/)  


### 12. [ECCV '2024 Oral] LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation
**Authors**: Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, Ziwei Liu
<details span>
<summary><b>Abstract</b></summary>
  3D content creation has achieved significant progress in terms of both quality and speed. Although current feed-forward models can produce 3D objects in seconds, their resolution is constrained by the intensive computation required during training. In this paper, we introduce Large Multi-View Gaussian Model (LGM), a novel framework designed to generate high-resolution 3D models from text prompts or single-view images. Our key insights are two-fold: 1) 3D Representation: We propose multi-view Gaussian features as an efficient yet powerful representation, which can then be fused together for differentiable rendering. 2) 3D Backbone: We present an asymmetric U-Net as a high-throughput backbone operating on multi-view images, which can be produced from text or single-view image input by leveraging multi-view diffusion models. Extensive experiments demonstrate the high fidelity and efficiency of our approach. Notably, we maintain the fast speed to generate 3D objects within 5 seconds while boosting the training resolution to 512, thereby achieving high-resolution 3D content generation.
</details>

  [📄 Paper](https://arxiv.org/pdf/2402.05054) | [💻 Code](https://github.com/3DTopia/LGM) | [🌐 Project Page](https://me.kiui.moe/lgm/)  



### 13. [Arxiv '2024] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting
**Authors**: Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, Zexiang Xu
<details span>
<summary><b>Abstract</b></summary>
  We propose GS-LRM, a scalable large reconstruction model that can predict high-quality 3D Gaussian primitives from 2-4 posed sparse images in 0.23 seconds on single A100 GPU. Our model features a very simple transformer-based architecture; we patchify input posed images, pass the concatenated multi-view image tokens through a sequence of transformer blocks, and decode final per-pixel Gaussian parameters directly from these tokens for differentiable rendering. In contrast to previous LRMs that can only reconstruct objects, by predicting per-pixel Gaussians, GS-LRM naturally handles scenes with large variations in scale and complexity. We show that our model can work on both object and scene captures by training it on Objaverse and RealEstate10K respectively. In both scenarios, the models outperform state-of-the-art baselines by a wide margin. We also demonstrate applications of our model in downstream 3D generation tasks.
</details>

  [📄 Paper](https://arxiv.org/pdf/2404.19702) | 💻 Code | [🌐 Project Page](https://sai-bi.github.io/project/gs-lrm/#BibTeX)  


### 14. [Arxiv '2024] LRM: Large Reconstruction Model for Single Image to 3D
**Authors**: Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, Hao Tan
<details span>
<summary><b>Abstract</b></summary>
  We propose the first Large Reconstruction Model (LRM) that predicts the 3D model of an object from a single input image within just 5 seconds. In contrast to many previous methods that are trained on small-scale datasets such as ShapeNet in a category-specific fashion, LRM adopts a highly scalable transformer-based architecture with 500 million learnable parameters to directly predict a neural radiance field (NeRF) from the input image. We train our model in an end-to-end manner on massive multi-view data containing around 1 million objects, including both synthetic renderings from Objaverse and real captures from MVImgNet. This combination of a high-capacity model and large-scale training data empowers our model to be highly generalizable and produce high-quality 3D reconstructions from various testing inputs, including real-world in-the-wild captures and images created by generative models.
</details>

  [📄 Paper](https://arxiv.org/pdf/2311.04400) | 💻 Code | [🌐 Project Page](https://yiconghong.me/LRM/)  


### 15. [Arxiv '2024] GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation
**Authors**: Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen, Ceyuan Yang, Sida Peng, Yujun Shen, Gordon Wetzstein
<details span>
<summary><b>Abstract</b></summary>
  We introduce GRM, a large-scale reconstructor capable of recovering a 3D asset from sparse-view images in around 0.1s. GRM is a feed-forward transformer-based model that efficiently incorporates multi-view information to translate the input pixels into pixel-aligned Gaussians, which are unprojected to create a set of densely distributed 3D Gaussians representing a scene. Together, our transformer architecture and the use of 3D Gaussians unlock a scalable and efficient reconstruction framework. Extensive experimental results demonstrate the superiority of our method over alternatives regarding both reconstruction quality and efficiency. We also showcase the potential of GRM in generative tasks, i.e., text-to-3D and image-to-3D, by integrating it with existing multi-view diffusion models.
</details>

  [📄 Paper](https://arxiv.org/pdf/2403.14621) | [💻 Code](https://github.com/justimyhxu/grm) | [🌐 Project Page](https://justimyhxu.github.io/projects/grm/)  


### 16. [Arxiv '2024] Instant3D: Fast Text-to-3D with Sparse-View Generation and Large Reconstruction Model
**Authors**: Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, Sai Bi
<details span>
<summary><b>Abstract</b></summary>
  Text-to-3D with diffusion models has achieved remarkable progress in recent years. However, existing methods either rely on score distillation-based optimization which suffer from slow inference, low diversity and Janus problems, or are feed-forward methods that generate low-quality results due to the scarcity of 3D training data. In this paper, we propose Instant3D, a novel method that generates high-quality and diverse 3D assets from text prompts in a feed-forward manner. We adopt a two-stage paradigm, which first generates a sparse set of four structured and consistent views from text in one shot with a fine-tuned 2D text-to-image diffusion model, and then directly regresses the NeRF from the generated images with a novel transformer-based sparse-view reconstructor. Through extensive experiments, we demonstrate that our method can generate diverse 3D assets of high visual quality within 20 seconds, which is two orders of magnitude faster than previous optimization-based methods that can take 1 to 10 hours.
</details>

  [📄 Paper](https://arxiv.org/pdf/2311.06214) | 💻 Code | [🌐 Project Page](https://jiahao.ai/instant3d/) 


### 17. [Arxiv '2024] Gamba: Marry Gaussian Splatting with Mamba for single view 3D reconstruction
**Authors**: Qiuhong Shen, Zike Wu, Xuanyu Yi, Pan Zhou, Hanwang Zhang, Shuicheng Yan, Xinchao Wang
<details span>
<summary><b>Abstract</b></summary>
  We tackle the challenge of efficiently reconstructing a 3D asset from a single image at millisecond speed. Existing methods for single-image 3D reconstruction are primarily based on Score Distillation Sampling (SDS) with Neural 3D representations. Despite promising results, these approaches encounter practical limitations due to lengthy optimizations and significant memory consumption. In this work, we introduce Gamba, an end-to-end 3D reconstruction model from a single-view image, emphasizing two main insights: (1) Efficient Backbone Design: introducing a Mamba-based GambaFormer network to model 3D Gaussian Splatting (3DGS) reconstruction as sequential prediction with linear scalability of token length, thereby accommodating a substantial number of Gaussians; (2) Robust Gaussian Constraints: deriving radial mask constraints from multi-view masks to eliminate the need for warmup supervision of 3D point clouds in training. We trained Gamba on Objaverse and assessed it against existing optimization-based and feed-forward 3D reconstruction approaches on the GSO Dataset, among which Gamba is the only end-to-end trained single-view reconstruction model with 3DGS. Experimental results demonstrate its competitive generation capabilities both qualitatively and quantitatively and highlight its remarkable speed: Gamba completes reconstruction within 0.05 seconds on a single NVIDIA A100 GPU, which is about 1,000× faster than optimization-based methods.
</details>

  [📄 Paper](https://arxiv.org/pdf/2403.18795) | [💻 Code](https://github.com/SkyworkAI/Gamba) | [🌐 Project Page](https://florinshen.github.io/gamba-project/) 


### 18. [Arxiv '2024] MVGamba: Unify 3D Content Generation as State Space Sequence Modeling
**Authors**: Xuanyu Yi, Zike Wu, Qiuhong Shen, Qingshan Xu, Pan Zhou, Joo-Hwee Lim, Shuicheng Yan, Xinchao Wang, Hanwang Zhang
<details span>
<summary><b>Abstract</b></summary>
  Recent 3D large reconstruction models (LRMs) can generate high-quality 3D content in sub-seconds by integrating multi-view diffusion models with scalable multi-view reconstructors. Current works further leverage 3D Gaussian Splatting as 3D representation for improved visual quality and rendering efficiency. However, we observe that existing Gaussian reconstruction models often suffer from multi-view inconsistency and blurred textures. We attribute this to the compromise of multi-view information propagation in favor of adopting powerful yet computationally intensive architectures (e.g., Transformers). To address this issue, we introduce MVGamba, a general and lightweight Gaussian reconstruction model featuring a multi-view Gaussian reconstructor based on the RNN-like State Space Model (SSM). Our Gaussian reconstructor propagates causal context containing multi-view information for cross-view self-refinement while generating a long sequence of Gaussians for fine-detail modeling with linear complexity. With off-the-shelf multi-view diffusion models integrated, MVGamba unifies 3D generation tasks from a single image, sparse images, or text prompts. Extensive experiments demonstrate that MVGamba outperforms state-of-the-art baselines in all 3D content generation scenarios with approximately only 0.1× of the model size.
</details>

  [📄 Paper](https://arxiv.org/pdf/2406.06367) | 💻 Code | 🌐 Project Page


### 19. [NIPS '2024] MVSplat360: Feed-Forward 360 Scene Synthesis from Sparse Views
**Authors**: Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang, Andrea Vedaldi, Tat-Jen Cham, Jianfei Cai
<details span>
<summary><b>Abstract</b></summary>
  We introduce MVSplat360, a feed-forward approach for 360° novel view synthesis (NVS) of diverse real-world scenes, using only sparse observations. This setting is inherently ill-posed due to minimal overlap among input views and insufficient visual information provided, making it challenging for conventional methods to achieve high-quality results. Our MVSplat360 addresses this by effectively combining geometry-aware 3D reconstruction with temporally consistent video generation. Specifically, it refactors a feed-forward 3D Gaussian Splatting (3DGS) model to render features directly into the latent space of a pre-trained Stable Video Diffusion (SVD) model, where these features then act as pose and visual cues to guide the denoising process and produce photorealistic 3D-consistent views. Our model is end-to-end trainable and supports rendering arbitrary views with as few as 5 sparse input views. To evaluate MVSplat360's performance, we introduce a new benchmark using the challenging DL3DV-10K dataset, where MVSplat360 achieves superior visual quality compared to state-of-the-art methods on wide-sweeping or even 360° NVS tasks. Experiments on the existing benchmark RealEstate10K also confirm the effectiveness of our model.
</details>

  [📄 Paper](https://arxiv.org/pdf/2411.04924) | [💻 Code](https://github.com/donydchen/mvsplat360) | [🌐 Project Page](https://donydchen.github.io/mvsplat360/) 


### 20. [ECCV '2024] latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction
**Authors**: Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele, Jan Eric Lenssen
<details span>ECCV
<summary><b>Abstract</b></summary>
  We present latentSplat, a method to predict semantic Gaussians in a 3D latent space that can be splatted and decoded by a light-weight generative 2D architecture. Existing methods for generalizable 3D reconstruction either do not scale to large scenes and resolutions, or are limited to interpolation of close input views. latentSplat combines the strengths of regression-based and generative approaches while being trained purely on readily available real video data. The core of our method are variational 3D Gaussians, a representation that efficiently encodes varying uncertainty within a latent space consisting of 3D feature Gaussians. From these Gaussians, specific instances can be sampled and rendered via efficient splatting and a fast, generative decoder. We show that latentSplat outperforms previous works in reconstruction quality and generalization, while being fast and scalable to high-resolution data.
</details>

  [📄 Paper](https://arxiv.org/pdf/2403.16292) | [💻 Code](https://github.com/Chrixtar/latentsplat) | [🌐 Project Page](https://geometric-rl.mpi-inf.mpg.de/latentsplat/) 



### 21. [ICCV '2023] FineRecon: Depth-aware Feed-forward Network for Detailed 3D Reconstruction
**Authors**: Noah Stier, Anurag Ranjan, Alex Colburn, Yajie Yan, Liang Yang, Fangchang Ma, Baptiste Angles
<details span>
<summary><b>Abstract</b></summary>
  Recent works on 3D reconstruction from posed images have demonstrated that direct inference of scene-level 3D geometry without test-time optimization is feasible using deep neural networks, showing remarkable promise and high efficiency. However, the reconstructed geometry, typically represented as a 3D truncated signed distance function (TSDF), is often coarse without fine geometric details. To address this problem, we propose three effective solutions for improving the fidelity of inference-based 3D reconstructions. We first present a resolution-agnostic TSDF supervision strategy to provide the network with a more accurate learning signal during training, avoiding the pitfalls of TSDF interpolation seen in previous work. We then introduce a depth guidance strategy using multi-view depth estimates to enhance the scene representation and recover more accurate surfaces. Finally, we develop a novel architecture for the final layers of the network, conditioning the output TSDF prediction on high-resolution image features in addition to coarse voxel features, enabling sharper reconstruction of fine details. Our method, FineRecon, produces smooth and highly accurate reconstructions, showing significant improvements across multiple depth and 3D reconstruction metrics.
</details>

  [📄 Paper](https://arxiv.org/pdf/2304.01480) | [💻 Code](https://github.com/apple/ml-finerecon) | 🌐 Project Page



### 22. [Arxiv '2024] Flex3D: Feed-Forward 3D Generation With Flexible Reconstruction Model And Input View Curation
**Authors**: Junlin Han, Jianyuan Wang, Andrea Vedaldi, Philip Torr, Filippos Kokkinos
<details span>
<summary><b>Abstract</b></summary>
  Generating high-quality 3D content from text, single images, or sparse view images remains a challenging task with broad applications. Existing methods typically employ multi-view diffusion models to synthesize multi-view images, followed by a feed-forward process for 3D reconstruction. However, these approaches are often constrained by a small and fixed number of input views, limiting their ability to capture diverse viewpoints and, even worse, leading to suboptimal generation results if the synthesized views are of poor quality. To address these limitations, we propose Flex3D, a novel two-stage framework capable of leveraging an arbitrary number of high-quality input views. The first stage consists of a candidate view generation and curation pipeline. We employ a fine-tuned multi-view image diffusion model and a video diffusion model to generate a pool of candidate views, enabling a rich representation of the target 3D object. Subsequently, a view selection pipeline filters these views based on quality and consistency, ensuring that only the high-quality and reliable views are used for reconstruction. In the second stage, the curated views are fed into a Flexible Reconstruction Model (FlexRM), built upon a transformer architecture that can effectively process an arbitrary number of inputs. FlemRM directly outputs 3D Gaussian points leveraging a tri-plane representation, enabling efficient and detailed 3D generation. Through extensive exploration of design and training strategies, we optimize FlexRM to achieve superior performance in both reconstruction and generation tasks. Our results demonstrate that Flex3D achieves state-of-the-art performance, with a user study winning rate of over 92% in 3D generation tasks when compared to several of the latest feed-forward 3D generative models.
</details>

  [📄 Paper](https://arxiv.org/pdf/2410.00890) | 💻 Code | [🌐 Project Page](https://junlinhan.github.io/projects/flex3d/) 



### 23. [Arxiv '2024] Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image
**Authors**: Stanislaw Szymanowicz, Eldar Insafutdinov, Chuanxia Zheng, Dylan Campbell, João F. Henriques, Christian Rupprecht, Andrea Vedaldi
<details span>
<summary><b>Abstract</b></summary>
  In this paper, we propose Flash3D, a method for scene reconstruction and novel view synthesis from a single image which is both very generalisable and efficient. For generalisability, we start from a "foundation" model for monocular depth estimation and extend it to a full 3D shape and appearance reconstructor. For efficiency, we base this extension on feed-forward Gaussian Splatting. Specifically, we predict a first layer of 3D Gaussians at the predicted depth, and then add additional layers of Gaussians that are offset in space, allowing the model to complete the reconstruction behind occlusions and truncations. Flash3D is very efficient, trainable on a single GPU in a day, and thus accessible to most researchers. It achieves state-of-the-art results when trained and tested on RealEstate10k. When transferred to unseen datasets like NYU it outperforms competitors by a large margin. More impressively, when transferred to KITTI, Flash3D achieves better PSNR than methods trained specifically on that dataset. In some instances, it even outperforms recent methods that use multiple views as input.
</details>

  [📄 Paper](https://arxiv.org/pdf/2406.04343) | [💻 Code](https://github.com/eldar/flash3d) | [🌐 Project Page](https://www.robots.ox.ac.uk/~vgg/research/flash3d/) 



### 24. [NIPS '2024] Epipolar-Free 3D Gaussian Splatting for Generalizable Novel View Synthesis
**Authors**: Zhiyuan Min, Yawei Luo, Jianwen Sun, Yi Yang
<details span>
<summary><b>Abstract</b></summary>
  Generalizable 3D Gaussian splitting (3DGS) can reconstruct new scenes from sparse-view observations in a feed-forward inference manner, eliminating the need for scene-specific retraining required in conventional 3DGS. However, existing methods rely heavily on epipolar priors, which can be unreliable in complex realworld scenes, particularly in non-overlapping and occluded regions. In this paper, we propose eFreeSplat, an efficient feed-forward 3DGS-based model for generalizable novel view synthesis that operates independently of epipolar line constraints. To enhance multiview feature extraction with 3D perception, we employ a selfsupervised Vision Transformer (ViT) with cross-view completion pre-training on large-scale datasets. Additionally, we introduce an Iterative Cross-view Gaussians Alignment method to ensure consistent depth scales across different views. Our eFreeSplat represents an innovative approach for generalizable novel view synthesis. Different from the existing pure geometry-free methods, eFreeSplat focuses more on achieving epipolar-free feature matching and encoding by providing 3D priors through cross-view pretraining. We evaluate eFreeSplat on wide-baseline novel view synthesis tasks using the RealEstate10K and ACID datasets. Extensive experiments demonstrate that eFreeSplat surpasses state-of-the-art baselines that rely on epipolar priors, achieving superior geometry reconstruction and novel view synthesis quality.
</details>

  [📄 Paper](https://arxiv.org/pdf/2410.22817) | 💻 Code | [🌐 Project Page](https://tatakai1.github.io/efreesplat/) 


### 25. [Arxiv '2024] No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images
**Authors**: Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, Songyou Peng
<details span>
<summary><b>Abstract</b></summary>
  We introduce NoPoSplat, a feed-forward model capable of reconstructing 3D scenes parameterized by 3D Gaussians from \textit{unposed} sparse multi-view images. Our model, trained exclusively with photometric loss, achieves real-time 3D Gaussian reconstruction during inference. To eliminate the need for accurate pose input during reconstruction, we anchor one input view's local camera coordinates as the canonical space and train the network to predict Gaussian primitives for all views within this space. This approach obviates the need to transform Gaussian primitives from local coordinates into a global coordinate system, thus avoiding errors associated with per-frame Gaussians and pose estimation. To resolve scale ambiguity, we design and compare various intrinsic embedding methods, ultimately opting to convert camera intrinsics into a token embedding and concatenate it with image tokens as input to the model, enabling accurate scene scale prediction. We utilize the reconstructed 3D Gaussians for novel view synthesis and pose estimation tasks and propose a two-stage coarse-to-fine pipeline for accurate pose estimation. Experimental results demonstrate that our pose-free approach can achieve superior novel view synthesis quality compared to pose-required methods, particularly in scenarios with limited input image overlap. For pose estimation, our method, trained without ground truth depth or explicit matching loss, significantly outperforms the state-of-the-art methods with substantial improvements. This work makes significant advances in pose-free generalizable 3D reconstruction and demonstrates its applicability to real-world scenarios.
</details>

  [📄 Paper](https://arxiv.org/pdf/2410.24207) | [💻 Code](https://github.com/cvg/NoPoSplat) | [🌐 Project Page](https://noposplat.github.io/) 



### 26. [CVPR'2024] MuRF: Multi-Baseline Radiance Fields
**Authors**: Haofei Xu, Anpei Chen, Yuedong Chen, Christos Sakaridis, Yulun Zhang, Marc Pollefeys, Andreas Geiger, Fisher Yu
<details span>
<summary><b>Abstract</b></summary>
  We present Multi-Baseline Radiance Fields (MuRF), a general feed-forward approach to solving sparse view synthesis under multiple different baseline settings (small and large baselines, and different number of input views). To render a target novel view, we discretize the 3D space into planes parallel to the target image plane, and accordingly construct a target view frustum volume. Such a target volume representation is spatially aligned with the target view, which effectively aggregates relevant information from the input views for high-quality rendering. It also facilitates subsequent radiance field regression with a convolutional network thanks to its axis-aligned nature. The 3D context modeled by the convolutional network enables our method to synthesis sharper scene structures than prior works. Our MuRF achieves state-of-the-art performance across multiple different baseline settings and diverse scenarios ranging from simple objects (DTU) to complex indoor and outdoor scenes (RealEstate10K and LLFF). We also show promising zero-shot generalization abilities on the Mip-NeRF 360 dataset, demonstrating the general applicability of MuRF.
</details>

  [📄 Paper](https://arxiv.org/pdf/2312.04565) | [💻 Code](https://github.com/autonomousvision/murf) | [🌐 Project Page](https://haofeixu.github.io/murf/) 


### 27. [3DV'2025] 3D Reconstruction with Spatial Memory
**Authors**: Hengyi Wang, Lourdes Agapito
<details span>
<summary><b>Abstract</b></summary>
  We present Spann3R, a novel approach for dense 3D reconstruction from ordered or unordered image collections. Built on the DUSt3R paradigm, Spann3R uses a transformer-based architecture to directly regress pointmaps from images without any prior knowledge of the scene or camera parameters. Unlike DUSt3R, which predicts per image-pair pointmaps each expressed in its local coordinate frame, Spann3R can predict per-image pointmaps expressed in a global coordinate system, thus eliminating the need for optimization-based global alignment. The key idea of Spann3R is to manage an external spatial memory that learns to keep track of all previous relevant 3D information. Spann3R then queries this spatial memory to predict the 3D structure of the next frame in a global coordinate system. Taking advantage of DUSt3R's pre-trained weights, and further fine-tuning on a subset of datasets, Spann3R shows competitive performance and generalization ability on various unseen datasets and can process ordered image collections in real time.
</details>

  [📄 Paper](https://arxiv.org/pdf/2408.16061) | [💻 Code](https://github.com/HengyiWang/spann3r) | [🌐 Project Page](https://hengyiwang.github.io/projects/spanner) 

### 26. [Arxiv'2025] Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass
**Authors**: Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, Matt Feiszli
<details span>
<summary><b>Abstract</b></summary>
  Multi-view 3D reconstruction remains a core challenge in computer vision, particularly in applications requiring accurate and scalable representations across diverse perspectives. Current leading methods such as DUSt3R employ a fundamentally pairwise approach, processing images in pairs and necessitating costly global alignment procedures to reconstruct from multiple views. In this work, we propose Fast 3D Reconstruction (Fast3R), a novel multi-view generalization to DUSt3R that achieves efficient and scalable 3D reconstruction by processing many views in parallel. Fast3R's Transformer-based architecture forwards N images in a single forward pass, bypassing the need for iterative alignment. Through extensive experiments on camera pose estimation and 3D reconstruction, Fast3R demonstrates state-of-the-art performance, with significant improvements in inference speed and reduced error accumulation. These results establish Fast3R as a robust alternative for multi-view applications, offering enhanced scalability without compromising reconstruction accuracy.
</details>

  [📄 Paper](https://arxiv.org/pdf/2501.13928) | 💻 Code | [🌐 Project Page](https://fast3r-3d.github.io/) 
