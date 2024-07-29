---
title:  "[arXiv preprint] Shape of Motion paper review"
excerpt: "Shape of Motion: 4D Reconstruction from a Single Video"

categories:
  - DL_paper
tags:
  - [DL, computer_vision]

published: true

toc: true
toc_sticky: true
 
date: 2024-07-29
last_modified_at: 2024-07-29
use_math: true
---

**[Title]** Shape of Motion: 4D Reconstruction from a Single Video

**[Keyword]** Human tracking, novel view synthesis, 3D Gaussian Splatting

**[Journal]** arXiv preprint arXiv:2407.13764

**[arXiv]** <a href="https://arxiv.org/abs/2407.13764" target="blank_">https://arxiv.org/abs/2407.13764</a>

**[Summary]**

<p align="center"><img src="https://github.com/user-attachments/assets/a6b51a46-4bf6-4255-a04a-e3fd2ada6c40"></p>

&nbsp;&nbsp;&nbsp;&nbsp;RGB video, Monodepth, and 2D Tracks(per-point) are given as the model's input. 2D tracks are lifted to 3D tracks by using a depth map, and 3D tracks are used to initialize the 3D Gaussians. Each Gaussians are clustered by velocity, and each clusters share the same rotation and translation. 3D Gaussians are used to represent the scene, and it synthetic the monodepth, 3D tracks, and RGB image by rasterization. The loss between 2D input and projected 2D output optimizes the parameter of 3D Gaussians.