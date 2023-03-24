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

# 0. Abstract

DETR은 obj detection에서 좋은 performance를 보여줌과 동시에 많은 hand-desinged componets를 제거함으로써 완전한 end-to-end의 학습을 할 수 있게 되었습니다. 그러나, Trnasformer attention module의 한계로 인해 DETR에는 다음과 같은 2가지 문제가 존재합니다.

1. **Slow convergence**

2. **Limited feature spatial resolution**

본 저자는 이러한 문제를 해결하기 위해 Deformable DETR을 제안합니다. Deformable DETR의 attention module은 reference point를 지정하여, 해당 point 근처에서만 small key sampling을 진행합니다. 이를 통해 기존 DETR의 문제를 상당부분 개선하였으며, small object에 대한 detection performance도 많이 향상시켰다고 합니다.

<br>
<br>

# 1. Introduction

## 1-1. DETR

### 1-1-1) About DETR

DETR 이전의 obj detection 모델은 NMS와 같은 hand-crafted components가 상당부분 존재했습니다. 그러나 2020년 등장한 DETR은, CNN과 Transformer의 encoder-decoder를 결합한 간단한 아키텍쳐를 도입하여 이러한 hand-crafted components를 모두 제거하였습니다. performance는 이전 모델들과 유사하게 유지하면서, Transformer의 강력한 relation modeling 능력을 통해 obj detection 분야에서 완전한 end-to-end를 구현한 최초의 모델이라고 할 수 있겠습니다.

### 1-1-2) Problem of DETR

DETR은 Abstract에서도 잠깐 언급했듯이 2가지의 문제점을 가지고 있습니다. <br>

1. **slow cpnvergence**

    우선, DETR은 수렴해에 도달하기까지의 시간이 매우 오래걸립니다. DETR은 COCO dataset에 대해 converge까지 약 500 에폭정도가 필요한데, 이는 Faseter R-CNN보다 10~20배 더 느린 수치입니다.

2. **small obj**

    DETR은 작은 오브젝트에 대해 매우 낮은 performance를 보여줍니다. DETR 이전의 모델들은 주로 output과 가까운 고해상도 feature map에서 small obj를 detect합니다. 하지만, DETR은 이러한 복잡한 feature map을 수용할 수 있는 능력이 없습니다.

정리해보자면, DETR의 attention module은 attention weight와 feature map의 모든 pixel을 cast하여 학습을 진행합니다. 따라서 긴 학습시간은 필수가결적이라고 할 수 있겠습니다. 또한, Transformer의 encoder는 모든 pixel에 대해 quadratic computation을 진행하게 되는데, 이는 매우 많은 연산량과 메모리를 필요로 합니다. 따라서 high-resolution feature map을 계산하는 데 한계가 존재하며, 이는 자연스럽게 small obj에 대한 performance 저하로 이어지게 됩니다.

## 1-2. Deformable DETR

![image](https://user-images.githubusercontent.com/84084372/227284261-54263268-e20b-4ad8-9742-71ca87e4ca2c.png)


본 저자는 Deformable Convolution(Dai et al., 2017)이라는 방법론을 도입함으로써 DETR의 느린 convergence issue와 high complexity issue를 해결합니다. 즉, Deformable DETR은 Transformer의 관계 모델링 능력과 deformable convolution의 soarse한 공간 샘플링 능력의 결합이라고 할 수 있겠으며, 이를 `deformable attention module`이라 명명하였습니다. Fig.1에서 확인할 수 있듯이 deformable attention module은 모든 픽셀을 attention weight와 match하지 않고, 특정 sampling location 주변의 pixel들만 사용합니다. 이를 통해 FPN의 도움 없이도 multi scale feature를 확장할 수 있었다고 합니다.

### 1-2-1) two-stage Deformable DETR

본 저자는 Deformable DETR의 빠른 convergence와 memory efficiency로 인해, `two-stage Deformable DETR`이라는 variants model을 시도해 볼 수 있었다고 말하고 있습니다. two-stage Deformable DETR은 첫 번쨰 stage에서 region proposal을 생성해내고, 이를 2번째 스테이지인 decoder에 넘겨줘서 iterative하게 bounding box refinement를 해주는 메커니즘을 가지고 있습니다. 즉, decoder에서는 region proposal을 받아 지속적으로 bounding box를 수정하면서 detection performance를 향상시킬 수 있었다고 합니다. 


### 1-2-2) Experiments

Deformable DETR은 기존 DETR보다 10배 더 적은 에폭을 사용하면서 더 좋은 performance를(특히 작은 물체에 대해서) 기록했다고 합니다. two-stage Deformable DETR을 사용하면 성능을 조금 더 향상시킬 수 있으며, 아래는 공식 깃헙 링크입니다.

> links : <a href="https://github.com/fundamentalvision/Deformable-DETR" target="_blank">https://github.com/fundamentalvision/Deformable-DETR</a>

<br>
<br>

# 3. Revisiting Transformers and DETR

## 3-1. Multi-Head Attention in Transformers

### 3-1-1) Intro

Transformer는 기계번역 분야에서 주로 사용되던 attention기반의 네트워크 아키텍쳐입니다. transformer는 input으로 query elements(e.g. output sentence의 target word)와 key elements(e.g. input sentence의 source word)를 받으며, query와 key element의 유사도를 측정하여 attention weight를 계산합니다. `multi-haed attention module`은 계산된 attention weight를 바탕으로 key contents를 adaptive하게 계산합니다. 이 때 모델은 여러개의 attention head를 가지며, 각각의 attention head는 서로 다른 subspace나 position으로부터 온 contents들을 인식합니다. 이러한 attention head들은 선형 결합을 통해 계산되며, 각각 학습 가능한 가중치를 가집니다. 계산식은 다음과 같이 정의될 수 있습니다.

$$
MultiHeadAttn(z_q, x) = \sum^M_{m=1}{W_m[\sum_{k \in \Omega_k}A_{mqk} \cdot W^{\prime}_mx_k]}
$$

- **Annotation**

  - $q \in \Omega_q$

    query element입니다. representation feature로 $z \in \mathbb{R}^C$를 가집니다.

  - $k \in \Omega_k$

    key element입니다. representation feature로 $x_k \in \mathbb{R}^C$를 가집니다.

  - $\Omega_q$ & $\Omega_k$

    각각 q와 k의 전체 집합을 의미합니다.

  - $m$

    attention head의 index를 의미합니다.

  - $Amqk$, $W_m$, $W^\prime_m$

    각각 attention weight, query weight, key weight를 의미합니다.

모든 key contents의 feature에 대해서 Attention weight와 key weight를 곱한 뒤 이를 전부 더하여 query weight와 곱하면, m번째 attention head에서의 feature가 계산됩니다. 또한, 위 수식에는 나와있지 않지만 query와 key elements의 feature간 공간 정보를 구별해주기 위해 position embedding을 사용합니다. 제가 attention 관련 논문을 읽어보지 않아서 key weight, query weight라는 표현이 맞는지는 모르겠는데, 각각 key와 query에 대해 고유하게 사용되는 weight이기 때문에 상기와 같이 표현하였다는 점 참고해주시면 감사하겠습니다.

### 3-1-2) Issue

Transformer에는 다음과 같은 2가지 issue가 존재합니다.

- **long training schedules**

  key의 개수($N_k$)가 매우 커지는 경우 attention weight는 $\frac{1}{N_k}$에 점점 수렴하게 되는데, 이렇게 되면 각 gradient의 차이가 모호해지기 때문에 각각 key들을 특정하기 위해 매우 긴 training time이 필요합니다. obj detection에 transformer를 사용하게 되면, key elements는 image의 pixel이 되며, 일반적으로 이미지의 픽셀 개수는 매우 많습니다 따라서 학습 시간이 길어지는 문제가 발생합니다.

- **computational and memory complexity**

  multi-head attention에서 시간 복잡도는 $O(N_qC^2 + N_kC^2 + N_qN_kC^2)$으로 표현될 수 있습니다. 이미지의 경우 dimension(C)에 비해 pixel 수($N_k$ = $N_q$)가 훨씬 많기 때문에 시간 복잡도는 $N_qN_kC^2$에 의해 지배됩니다. 따라서 feature map의 size가 증가함에 따라 복잡도는 quadratic하게 증가하며, 이는 계산 및 메모리 복잡도를 증가시킵니다.

<br>
<br>

# 4. Method

## 4-1. Deformable Transformers for End-to-End Object Detection

### 4-1-1) Deformable Attention Module

![image](https://user-images.githubusercontent.com/84084372/227588961-b03a34af-fc7c-4e04-a757-a4f101983c9f.png)


이전 연구에서 transformer를 image에 적용하고자 하는 시도는, transformer가 image feature map의 모든 pixel을 고려하기 때문에 메모리 및 학습 속도 이슈를 발생시켰습니다. 본 저자는 이러한 문제를 다루기 위해 `deformable attention module`을 도입합니다.