---
title:  "Deformable DETR(2021) 논문 정리 (진행중)"
excerpt: "DEFORMABLE DETR: DEFORMABLE TRANSFORMERS
FOR END-TO-END OBJECT DETECTION"

categories:
  - DL_paper
tags:
  - [DL, computer_vision]

published: true

toc: true
toc_sticky: true
 
date: 2023-03-24
last_modified_at: 2022-03-24
use_math: true
---

# Abstract

DETR은 obj detection에서 좋은 performance를 보여줌과 동시에 많은 hand-desinged componets를 제거함으로써 완전한 end-to-end의 학습을 할 수 있게 되었습니다. 그러나, Trnasformer attention module의 한계로 인해 DETR에는 다음과 같은 2가지 문제가 존재합니다.

1. Slow convergence

2. Limited feature spatial resolution

본 저자는 이러한 문제를 해결하기 위해 Deformable DETR을 제안합니다. Deformable DETR의 attention module은 reference point를 지정하여, 해당 point 근처에서만 small key sampling을 진행합니다. 이를 통해 기존 DETR의 문제를 상당부분 개선하였으며, small object에 대한 detection performance도 많이 향상시켰다고 합니다.

# Introduction

DETR 이전의 obj detection 모델은 NMS와 같은 hand-crafted components가 상당부분 존재했습니다. 그러나 2020년 등장한 DETR은, CNN과 Transformer의 encoder-decoder를 결합한 간단한 아키텍쳐를 도입하여 이러한 hand-crafted components를 모두 제거하였습니다. performance는 이전 모델들과 유사하게 유지하면서, Transformer의 강력한 relation modeling 능력을 통해 obj detection 분야에서 완전한 end-to-end를 구현한 최초의 모델이라고 할 수 있겠습니다.

detr 문제점
1) slow cpnvergence
2) small obj x