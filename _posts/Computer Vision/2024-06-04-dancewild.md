---
title:  "[3DV 2021] Dance In the Wild 논문 리뷰"
excerpt: "Dance In the Wild: Monocular Human Animation with Neural Dynamic Appearance Synthesis"

categories:
  - DL_paper
tags:
  - [DL, computer_vision]

published: true

toc: true
toc_sticky: true
 
date: 2024-06-04
last_modified_at: 2024-06-04
use_math: true
---

> 논문링크 : <a href="https://arxiv.org/pdf/2111.05916" target="blank_">Dance In the Wild: Monocular Human Animation with Neural Dynamic Appearance Synthesis</a>

> Project Page : <a href="https://geometry.cs.ucl.ac.uk/projects/2021/danceinwild/" target="blank_">https://geometry.cs.ucl.ac.uk/projects/2021/danceinwild/</a>

오늘 소개드릴 페이퍼는 CVPR 2024에 accept된 `Dance In the Wild: Monocular Human Animation with Neural Dynamic Appearance Synthesis`입니다.

<br>
<br>

## 1. Abstract & Introduction

&nbsp;&nbsp;&nbsp;&nbsp;Human motion을 generation 하는 것은 AR/VR 분야에서 중요한 역할을 합니다. 하지만, 당시 발표된 논문들은 dance와 같은 high dynamic motion, 나풀거리는 옷 등에 대해 제대로 된 generation을 하지 못했다고 합니다. 따라서 해당 페이퍼의 저자들은 이러한 당시 문제점들을 해결하기 위해 `이전 10 프레임에 대한 모션 정보를 사용`했다고 합니다.

&nbsp;&nbsp;&nbsp;&nbsp;이전 프레임의 모션 정보가 선행 연구의 문제점을 해결할 수 있었던 이유는 다음과 같습니다. 우선, 옷의 움직임은 사람의 모션에 의해 크게 영향을 받습니다. 따라서 옷의 움직임을 생성할 때, motion 정보를 condition으로 주게 되면 generate performance에 크게 도움을 준다고 합니다. 또한, motion 정보는 pose를 refine 하는데에도 크게 도움을 준다고 합니다. 자세한 내용은 다음 섹션에서 설명드리도록 하겠습니다.

<br>
<br>

## 2. Method

### 2-1. Overview

<p align="center"><img src="https://github.com/On-JungWoan/paper-review/assets/84084372/490c06b5-d665-445a-8184-3582fe740f1f"></p>

&nbsp;&nbsp;&nbsp;&nbsp;전체적인 method는 위와 같습니다. Input으로는 Human body dense UV map과 sketon을 받으며, StyleGAN을 사용하여 이에 해당하는 human motion을 생성합니다. 해당 페이퍼의 key contribution은 다음과 같습니다.

1. video 속 actor가 나풀거리는 옷을 입고 있을 경우 해당 outfit은 motion의 영향을 크게 받음.
    - 나풀거리는 옷의 움직임을 정확하게 생성하는 것은 dance generation에 있어서 challenging task.
    - 따라서 **motion signature를 AdaIN operator의 input**으로 넣어주면 generate performance에 크게 도움을 줄 수 있음
        - 옷의 움직임이 motion의 영향을 크게 받기 때문.
        - AdaIN이란? style transfer에 사용되는 전통적인 operator.
2. pose estimator는 single image를 기반으로 학습되었기 때문에 self-occlusion 및 복잡한 pose에서의 incorrect estimation 문제가 있으며, jittering이 심하게 나타날 수 있음.
    - 해당 문제를 해결하는 데에 이전 frame 정보들이 강력한 cue가 될 수 있음
    - 따라서 **이전 frame의 모션 정보를 활용하여 pose를 refine** 함.

<br>

### 2-2. Input

&nbsp;&nbsp;&nbsp;&nbsp;먼저, 모델의 input에 대해 설명드리도록 하겠습니다. 우선, video로부터 RGB image sequence $\{ \mathbb{I}_i \}$를 추출합니다.

$$
\{ \mathbb{I}_i \}, \; i=1...N
$$

<br>
  
&nbsp;&nbsp;&nbsp;&nbsp;이후, input image에 대해 dense body UV map과 keypoint를 추출합니다. 이를 위한 off-the-shelf model로는 각각 DensePose와 OpenPose를 사용합니다.

$$
IUV \; maps \in \mathbb{R}^{3 \times W \times H} \\
keypoints \in \mathbb{R}^{3 \times W \times H}
$$

<br>
  
&nbsp;&nbsp;&nbsp;&nbsp;추출한 UV map과 keypoint를 concat하면 `첫 번째 input인 Pose Signature` $\mathbb{P}_i$가 됩니다. (3,W,H)의 matrix 2개를 concat하였기 때문에 Pose Signature는 (6, W, H)의 dimension을 갖습니다.

$$
\mathbb{P}_i = IUV \; maps \oplus keypoints \\
\mathbb{P}_i \in \mathbb{R}^{6\times W \times H}
$$

<br>

&nbsp;&nbsp;&nbsp;&nbsp;위 pose signature를 이전 K개 frame에 대해 동일하게 샘플링 한 뒤 concat한 것이 `두 번째 input인 Motion Signature` $\mathbb{M}_i$입니다. 이 때, K값으로는 10을 사용했으며, 해당 K frame은 이전 20개 frame에서 랜덤하게 샘플링됩니다. 따라서 Motion signature는 (6*K,W,H)의 dim을 가지게 됩니다.

$$
\mathbb{M}_i \in \mathbb{R}^{60 \times W \times H}, \quad K=10
$$

<br>

&nbsp;&nbsp;&nbsp;&nbsp;지금까지의 일련의 과정을 거치고 난 뒤, pose signature $\mathbb{P}_i$와 motion signature $\mathbb{M}_i$가 모델의 input으로 들어가게 됩니다.
    
<br>

### 2-3. Learning Pose/Motion Features

- **2D spatial pose features**

  &nbsp;&nbsp;&nbsp;&nbsp;첫 번째 input인 Pose signature $\mathcal{P}_i$는 CNN based의 pose encoder $E_P$의 input으로 들어가, pose feature $P_i$를 계산하는 데 사용됩니다. 이 때, out feature의 dimension은 512이며 spatial dim은 기존의 1/16로 감소합니다.

$$
\mathcal{P}_i \in \mathbb{R}^{512 \times W_s \times H_s} \\
(where \; W_s=W/16, \; H_s=H/16)
$$

<br>

- **1D motion feature**

  &nbsp;&nbsp;&nbsp;&nbsp;두 번째 input인 Motion signature $\mathcal{M}_i$ 역시 Motion Encoder $E_M$의 input으로 들어가, Motion feature $M_i$를 계산하는 데 사용됩니다. 해당 feature는 2048 dim을 갖는 1D vector입니다.

$$
\mathcal{M}_i \in \mathbb{R}^{2048}
$$

<br>

- **Temporal coherent refinement**

  &nbsp;&nbsp;&nbsp;&nbsp;앞서 설명드렸듯이, 이전 프레임의 모션 정보 $M_i$는 pose feature $P_i$를 refine하는 데 사용됩니다. 이를 위해 motion feature를 모든 spatial chanel에 더해준 뒤, 해당 matrix를 refine network $E_{Refine}$의 input으로 넣어 refined pose $\tilde{\mathcal{P}_i}$를 계산합니다. 수식적으로는 다음과 같이 표현할 수 있습니다.

  - **step1) Reshape motion feature**

    (2048) → (2048, 1, 1) → (2048, $W_s$, $H_s$)

  - **step2) Concat**

    <p align="center"><img src="https://github.com/On-JungWoan/paper-review/assets/84084372/c5920358-658a-4d02-964d-34359a38c357"></p>

  - **step3) Refine**

    <p align="center"><img src="https://github.com/On-JungWoan/paper-review/assets/84084372/c5920358-658a-4d02-964d-34359a38c357"></p>
    
<br>

### 2-4. Style-based Generator

&nbsp;&nbsp;&nbsp;&nbsp;또한, Motion Signature $\mathcal{M}_i$는 옷의 움직임을 생성하는 데에도 사용됩니다. 이를 위해 Motion Signature를 StyleGAN $T$의 condition으로 넣어 옷에 대한 generate performance를 향상하였다고 합니다. 따라서 최종적으로 StyleGAN은 Refined pose $\tilde{\mathcal{P}_i}$와 Motion Signature $\mathcal{M}_i$를 input으로 받아 이미지 $\mathbb{I}^{\prime}_i$을 생성하게 됩니다.

$$
T(\tilde{\mathcal{P}_i}, \; \mathcal{M}_i) = \mathbb{I}^{\prime}_i
$$

<br>

&nbsp;&nbsp;&nbsp;&nbsp;다음은 지금까지의 모든 과정을 수식으로 표현한 것입니다. Pose signature와 Motion signature가 각각의 encoder로 들어가서 feature를 뽑는 데 사용되고, 해당 feature는 서로 concat되어 refine network의 input으로 다시 들어가게 됩니다. 그리고 refine된 pose feature는 다시 Motion Signature와 함께 GAN의 input으로 들어가 이미지를 생성하게 됩니다.

<p align="center"><img src="https://github.com/On-JungWoan/paper-review/assets/84084372/bb05fe90-2c53-48dc-923e-b533f63dce38"></p>

<br>

### 2-5. Loss

&nbsp;&nbsp;&nbsp;&nbsp;Loss는 크게 L1 reconstruction loss, perceptual loss, Adversarial loss로 나누어져 있습니다. 각 Loss가 의미하는 바는 다음과 같습니다.

<p align="center"><img src="https://github.com/On-JungWoan/paper-review/assets/84084372/d5c1fb33-7955-447e-87d0-d8dd2471ab13"></p>

<br>

- **L1 reconstruction loss**

  <p align="center"><img src="https://github.com/On-JungWoan/paper-review/assets/84084372/4630c6f7-3e4b-4821-a858-8716f5a768a6"></p>

  생성된 이미지와 GT 이미지간의 L1 loss

<br>

- **VGG-based perceptual loss**

  <p align="center"><img src="https://github.com/On-JungWoan/paper-review/assets/84084372/7da35182-a1e6-46c3-9173-d30c8f677475"></p>

  ImageNet으로 pre-trained된 VGG의 k번째 layer를 사용하여 perceptual loss를 계산

<br>

- **Adversarial Loss**

  <p align="center"><img src="https://github.com/On-JungWoan/paper-review/assets/84084372/7816c1c1-509b-4715-a129-e1fa335b5140"></p>

  GAN에서 사용하는 adv. loss와 동일

<br>
<br>

## 3. Experiments

### 3-1. Result
    
<p align="center"><img src="https://github.com/On-JungWoan/paper-review/assets/84084372/d8775901-44f7-40c2-b076-b3b969a09901"></p>

<p align="center"><img src="https://github.com/On-JungWoan/paper-review/assets/84084372/3298a96a-e191-409c-867e-753ae7aaf038"></p>
    

<br>

### 3-2. Motion signature

<p align="center"><img src="https://github.com/On-JungWoan/paper-review/assets/84084372/bc0ce1a5-06b3-4554-83f5-370af3ee2fa4"></p>

motion signature와 옷의 움직임간의 상관관계를 검증한 실험입니다.

- **Forward motion**
  - 정상적인 motion signature를 사용했을 때. → 치마가 움직이는 방향이 GT와 동일
- **Frozen motion**
  - K개의 동일한 frame을 사용하여 motion signature를 만듦 → 치마가 멈춘듯한 효과
- **Backward motion**
  - motion signature의 frame을 reverse 시킴 → 생성된 모션에 등장하는 치마의 방향 역시 반대 방향으로 바뀜

<br>

### 3-3. Pose refine

아래 figure는 100 프레임의 비디오에 대한 vertical & horizontal pixel들을 잘라서 이어붙인 결과입니다. motion signature를 사용했을 때(Ours), pose refine을 하지 않은 모델(Vanilla) 대비 jittering이 완화되는 것을 확인할 수 있습니다.

<p align="center"><img src="https://github.com/On-JungWoan/paper-review/assets/84084372/13401d3c-76d1-4fff-acb0-a357d7f0fd5c"></p>

<br>

### 3-4. **motion transfer**

특정 Actor의 motion(흰색 원피스)을 다른 actor에게 transfer하는 것도 가능하다고 합니다.
    
<p align="center"><img src="https://github.com/On-JungWoan/paper-review/assets/84084372/f38fa932-f2ad-4319-b0e0-b688607cb806"></p>