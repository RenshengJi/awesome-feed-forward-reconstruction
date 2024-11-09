# awesome-feed-forward-reconstruction

A curated list of papers and open-source resources focused on reconstruction based on large feed-forward model. 

### 1. [NIPS '24] SCube: Instant Large-Scale Scene Reconstruction using VoxSplats
**Authors**: Xuanchi Ren, Yifan Lu, Hanxue Liang, Zhangjie Wu, Huan Ling, Mike Chen, Sanja Fidler, Francis Williams, Jiahui Huang
<details span>
<summary><b>Abstract</b></summary>
  We present SCube, a novel method for reconstructing large-scale 3D scenes (geometry, appearance, and semantics) from a sparse set of posed images. Our method encodes reconstructed scenes using a novel representation VoxSplat, which is a set of 3D Gaussians supported on a high-resolution sparse-voxel scaffold. To reconstruct a VoxSplat from images, we employ a hierarchical voxel latent diffusion model conditioned on the input images followed by a feedforward appearance prediction model. The diffusion model generates high-resolution grids progressively in a coarse-to-fine manner, and the appearance network predicts a set of Gaussians within each voxel. From as few as 3 non-overlapping input images, SCube can generate millions of Gaussians with a 1024^3 voxel grid spanning hundreds of meters in 20 seconds. Past works tackling scene reconstruction from images either rely on per-scene optimization and fail to reconstruct the scene away from input views (thus requiring dense view coverage as input) or leverage geometric priors based on low-resolution models, which produce blurry results. In contrast, SCube leverages high-resolution sparse networks and produces sharp outputs from few views. We show the superiority of SCube compared to prior art using the Waymo self-driving dataset on 3D reconstruction and demonstrate its applications, such as LiDAR simulation and text-to-scene generation.
</details>

  [📄 Paper](https://arxiv.org/pdf/2410.20030) | [💻 Code]([https://github.com/hyzhou404/HUGS](https://github.com/nv-tlabs/SCube)) | [🌐 Project Page](https://research.nvidia.com/labs/toronto-ai/scube/)  