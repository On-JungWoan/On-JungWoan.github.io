---
title:  "[private] SpatialVLM(2024) 논문 리뷰"
excerpt: "Spatial VLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities"

categories:
  - DL_paper
tags:
  - [DL, computer_vision]

published: true

toc: true
toc_sticky: true
 
date: 2024-02-21
last_modified_at: 2024-02-21
use_math: true
---

> 논문링크 : <a href="https://arxiv.org/pdf/2401.12168.pdf" target="blank_">Spatial VLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities</a>

> project page : <a href="https://spatial-vlm.github.io/" target="blank_">https://spatial-vlm.github.io/</a>

구글 딥마인드에서 2024년 1월에 publish한 논문입니다.

<br>
<br>

# 1. Abstract

<img width="840" alt="image" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/717fffd8-0c13-4b1f-b09b-40299272112a">

&nbsp;&nbsp;&nbsp;&nbsp;VQA와 robotics 분야에 있어서 spatial relationship에 대해 이해하는 것은 상당히 중요합니다. CLIP등으로 대표되는 VLM이 최근 VQA 벤치마크에서 주목할만한 성능을 보였으나, 여전히 3D space에 대한 이해능력은 많이 뒤떨어지는 실정입니다. 위 figure에서 언급된 바와 같이, 모델은 input image에서 baseball player와 black man 사이의 거리를 인지하지 못합니다. 저자는 이러한 한계의 원인이 3D spatial knowledge의 부족에 있다고 하였으며, 이러한 문제를 인터넷에서 수집된 데이터만을 사용하여 해결하려는 것이 문제라고 밝히고 있습니다.

&nbsp;&nbsp;&nbsp;&nbsp;해당 연구에서는, 이러한 문제를 해결하기 위해 3D spatial VQA 데이터를 automatic하게 generation하는 framework를 개발하였으며, 그 결과 20억개의 VQA example을 1000만개의 real-world image로 scale up하는 데 성공했다고 합니다. 이러한 데이터셋을 사용하여 VLM을 훈련함으로써, spatial VQA에 대한 질적/양적인 성능을 상당히 끌어올릴 수 있었으며, spatial reasoning과 robotics 분야로의 새로운 downstream applications을 가능케하였다고 합니다.

<br>
<br>

# 2. Introduction

&nbsp;&nbsp;&nbsp;&nbsp;최근 image captioning, VQA, action recognition 등의 다양한 분야에서, VLM이 상당한 두각을 나타내고 있습니다. 하지만 VLM의 눈부신 성장과는 대조적으로, VLM이 고질적으로 겪고있는 문제가 있습니다. 그것은 바로 SOTA VLM들의 spatial 정보에 대한 이해 부족입니다. 가령, 3D space상의 다양한 object들의 spatial한 관계라던지, 해당 object의 position에 대한 이해를 필요로 하는 task에서는 VLM이 상당히 약한 모습을 보이고 있습니다. 이러한 spatial information을 이해하는 능력은 그 자체로도 매우 쓸모있을 뿐만 아니라, robotics나 AR등의 downstream task에서도 유용하게 사용될 수 있습니다. 따라서 이러한 limitation을 해결하는 것이 VLM에 있어서 가장 중요한 도전과제라고 할 수 있습니다.