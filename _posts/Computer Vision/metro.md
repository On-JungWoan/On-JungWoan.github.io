---
title:  "-"
excerpt: "-"

categories:
  - DL_paper
tags:
  - [DL, computer_vision]

published: true

toc: true
toc_sticky: true
 
date: 2023-07-13
last_modified_at: 2022-07-13
use_math: true
---

> 발표자료 : <a href="https://docs.google.com/presentation/d/1vXGcHwAJxjAXV_m76dgl7abqPDun0TY_/edit?usp=sharing&ouid=116507288704586191771&rtpof=true&sd=true" target="blank_">https://docs.google.com/presentation/d/1vXGcHwAJxjAXV_m76dgl7abqPDun0TY_/edit?usp=sharing&ouid=116507288704586191771&rtpof=true&sd=true</a>

> 논문링크 : <a href="https://arxiv.org/abs/1912.05656" target="blank_">VIBE: Video Inference for Human Body Pose and Shape Estimation</a>

> Implementation : <a href="https://github.com/mkocabas/VIBE" target="blank_">https://github.com/mkocabas/VIBE</a>

# Abstract

METRO -> single 이미지로부터 3D human pose와 mesh vertices를 reconstruction한다.
vertex-vertex & vertex-joint간의 관계를 학습하기 위해 transformer encoder 사용
output은 3D joint 좌표와 mesh vertices를 동시에 출력

METRO는 SMPL과 같은 parametric mesh model에 의존하지 않는다 -> hand와 같은 다른 도메인으로 쉽게 확장 가능
mesh topology를 완화 & 자유롭게 아무 vertice끼리 self-attetnion 가능하게 함으로써 non-local 관계 학습
Masked Vertex Modeling을 사용 -> mesh reconstruction 분야에서 매우 challening한 occlusion에 robust해짐


human mesh reconsturction(Human3.6M, 3DPW)에서 SOTA 달성
뿐만아니라  3D hand reconstruction 분야에서도 SOTA를 달성함 (FreiHAND)
-> 다른 도메인으로 쉽게 확장될 수 있는 특징

# Introduction

3D human pose & mesh reconstruction -> 많은 분야에 응용될 수 있는 매력적인 연구
but 복잡한 모션과 occlusion등은 여전히 challenging한 문제

최근 연구는 rough하게 2개의 카테고리로 나눌 수 있음

- Use parametric model
  - SMPL로 대두되며, shape과 pose coefficient를 predict하도록 학습됨
  - 많은 연구들이 parametric model을 사용하며, parametric model에 encode되어 있는 the strong prior덕에 다양한 환경에서도 robust하다.
  - 그러나 pose와 shape space는 한정되어 있음 model을 구축할 때 사용된 제한된 sample로 인해서

- Don't use parametric model
  - 이러한 parametric model의 한계를 극복하기 위함
  - 대표적인 방법으로는 다음과 같음
    - GCNN (Graph Convolutional Neural Network)
      - 이웃한 vertex-vertex간의 관계를 모델링
    - 1D heatmap
      - vertex 좌표를 regression
  - but, 이러한 방법은 non-local한 vertex끼리의 관계를 잘 학습하지 못함.

최근 연구에서 손과 발처럼 non-local한 vertices들 사이에 강한 correlation이 존재한다는 것을 밝혔음
따라서 저자들은 거리에 관계없이 joint 혹은 vertice들의 global한 관계를 학습하는 것이 occlusion등의 challenge를 해결하는 데 도움이 될 것이라고 주장.
따라서 METRO는 이러한 global한 interaction을 학습하기 위해 attention 연산을 사용하는 transformer를 key idea로 삼음

~transformer 강점 설명~

# contribution
  - Transformer encoder를 3D human pose and mesh 분야에 최초로 적용한 연구
  - 이를 통해 
  - MVM (Masked Vertex Modeling) 기법을 사용, occlusion등의 상황에 모델이 좀 더 robust해짐
  - 3DPW, Human3.6M dataset과 같은 human pose뿐만 아니라 FreiHAND와 같은 Hand 분야에서도 SOTA를 달성함 -> 다른 도메인으로의 확장성이 높음

# Method

## architecture

input으로 224x224의 single 이미지를 받아, body joint와 mesh vertice를 동시에 predict합니다.
제안된 프레임워크는 CNN과 Multi-Layer Transformer Encoder로 이루어져 있음.
CNN : input image로부터 feature map 추출
Multi-Layer Transformer Encoder : input으로 feature vector를 받아 joint와 vertex의 3D 좌표를 병렬적으로 출력함

## Convolutional Neural Network

ImageNet으로 pre-train된 모델 사용
output feature map의 dimension은 일반적으로 사용하는 2048
but, 특이하게 FC Layer를 거치지 않고 마지막 hidden layer에서 feature vector를 추출
그래서 output 보면 batchx2048x7x7
why? resolution이 높은 feature vector를 사용하는 것이 transformer의 performance를 높이는 데 도움을 줌.

따라서 HRNet과 같이 feature map의 resolution이 높은 large scale CNN을 사용해서 transformer의 performance를 더 끌어올렸다고 합니다.
이에 대한 ablation study는 뒤에서 더 자세히 다루도록 하겠습니다.

## Multi-Layer Transformer Encoder with Progressive Dimensionality Reduction

METRO의 최종 output은 3D 좌표이기 때문에 channel을 3으로 맞추어 줘야 함.
but backbone에서 FC layer를 거치지 않았기 때문에 encoder로 들어오는 차원은 2048임.
반면, original Transformer Encoder는 constant한 dimension을 가짐.
따라서 Encoder layer를 여러개 쌓고 각 layer를 통과할 때 마다 linear layer를 거치게 함으로써 차원을 줄여나감.
fig2와 같이 점점 줄여나가서 결국 3 dimension에 맞춤. -> FC Layer와 비슷한 역할을 한다고 생각하면 될듯