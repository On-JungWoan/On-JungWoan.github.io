---
title:  "A2J-Transformer(2023) 논문 리뷰"
excerpt: "A2J-Transformer: Anchor-to-Joint Transformer Network for 3D Interacting
Hand Pose Estimation from a Single RGB Image"

categories:
  - DL_paper
tags:
  - [DL, computer_vision]

published: true

toc: true
toc_sticky: true
 
date: 2023-07-18
last_modified_at: 2023-07-18
use_math: true
---

{% capture paper_name %}A2J-Transformer: Anchor-to-Joint Transformer Network for 3D Interacting
Hand Pose Estimation from a Single RGB Image{% endcapture %}
{% capture ppt_link %}https://docs.google.com/presentation/d/1XPV-8VHbJFCWYn86vBiAoeknDPXMD6G4/edit?usp=sharing&ouid=116507288704586191771&rtpof=true&sd=true{% endcapture %}
{% capture paper_link %}https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_A2J-Transformer_Anchor-to-Joint_Transformer_Network_for_3D_Interacting_Hand_Pose_Estimation_CVPR_2023_paper.pdf{% endcapture %}
{% capture github_link %}https://github.com/ChanglongJiangGit/A2J-Transformer{% endcapture %}

> 발표자료 :  <a href="{{ ppt_link }}" target="blank_">{{ ppt_link }}</a>

> 논문링크 : <a href="{{ paper_link }}" target="blank_">{{ paper_name }}</a>

> Implementation : <a href="{{ github_link }}" target="blank_">{{ github_link }}</a>

# 0. Abstract

single RGB image에서 3D interaction hand pose estimation을 하는 것은 매우 challenging한 task임.
- why?
  - self/inter occlusion
  - 양 손의 생김새가 비슷함
  - joint를 2D to 3D로 mapping할 잘못 mapping 됨
이러한 문제를 해결하기 위해 저자는 기존 A2J를 RGB domain으로 확장함

key idea는 다음과 같음
joint끼리의 global한 articulated clue와 interacting hand의 미세한 detail을 잘 capture하기 위해 A2J + transfomer 합침

A2J에 transformer를 합침으로써 A2J보다 좋은 점?
1. local anchor point들에 self attention을 걸어줌으로써 anchor들이 global한 joint의 정보를 학습하고, occlusion에 robust해짐
2. 각 anchor point는 모두 동일한 local representation을 갖는 것이 아니라 learnable한 query로 학습 됨. 이를 통해 pattern fitting capacity를 촉진시킴
3. A2J와는 다르게 anchor point들은 2D 공간이 아닌 3D 공간상에 위치하여 3D pose를 prediction함.

A2J-Transformer는 InterHand 2.6M에서 SOTA를 달성함. 그리고 강력한 generalization으로 인해 depth 도메인에서도 사용가능함.

- **Reference**

  [The Reprojection Error?](https://www.camcalib.io/post/what-is-the-reprojection-error)

  [Camera Matrix](https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf)

  [Camera Matrix](https://darkpgmr.tistory.com/32)

  [카메라 캘리브레이션 (Camera Calibration)](https://heekangpark.github.io/ml-shorts/positional-encoding-vs-positional-embedding)

  [Basic CNN Architecture: Explaining 5 Layers of Convolutional Neural Network](https://www.upgrad.com/blog/basic-cnn-architecture/)


  [트랜스포머(Transformer) 파헤치기—1. Positional Encoding](https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding)
