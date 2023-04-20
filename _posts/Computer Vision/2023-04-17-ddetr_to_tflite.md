---
title:  "Deformable DETR for edge device(초안)"
excerpt: "Convert Deformable DETR to TFLite"

categories:
  - DL_optim
tags:
  - [DL, computer_vision]

published: true

toc: true
toc_sticky: true
 
date: 2023-04-18
last_modified_at: 2023-04-18
use_math: true
---

최근 발표된 SOTA 모델의 경우, 대부분 그 크기가 매우 크고 무겁습니다(특히 비전분야). 이러한 pre-trained 모델들을 개발 환경에서 사용할 때는 대부분 큰 문제가 되지 않습니다. 하지만 Android나 임베디드 보드와 같은 엣지 디바이스에서는 이러한 Full-size 모델을 사용하는 데 한계가 존재합니다. 따라서 torch 모델을 엣지 디바이스에서 사용하기 위해서는, 이를 최적화 된 포맷으로 변환해주는 작업이 필요합니다. 해당 포스팅에서 사용할 torch 모델은 ICLR 2021에서 발표된 `Deformable DETR`이라는 Transformer based의 Object detection 모델이며, 이 모델을 엣지 디바이스에 최적화가 아주 잘 되어있는 `TFLite`로 변환하는 과정에 대해서 소개해드리도록 하겠습니다.

# 1. Conversion process

![image](https://user-images.githubusercontent.com/84084372/232539845-4c3c2d56-4c80-4fff-8037-31ced3fedaa3.png)


# 1. Changes to the torch model

