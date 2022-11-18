---
title:  "TensorRT Engine 성능 비교"
excerpt: "쓰레기 무단투기 탐지 프로젝트"

categories:
  - cctv
tags:
  - [딥러닝, computer_vision, ICT인턴십, 유클리드소프트]

published: true

toc: true
toc_sticky: true
 
date: 2022-10-14
last_modified_at: 2022-10-14
til: 'true'
---

## 1. Make TensorRT Engine
기존 TensorRT 모델에서 engine 모듈을 분리하는데 성공하였다. 자세한 개념은 아래에 정리해두었다. 

- **Ref.**

  [TensorRT 모델에서 engine 모듈 추출하기](https://on-jungwoan.github.io/dl/tensorrtengine/)


<br>
<br>

## 2. 성능 비교

다음은 Pytorch 모델과 TorchRT 모델, TorchRT Engine 파일의 성능 및 결과를 비교해본 결과이다. 성능은 1초당 연산한 횟수를 기준으로 하였으며 tqdm을 사용하였다. 해당 모델은 4개의 모델을 Blending 시켜놓은 멀티모델이므로, 모델당 2개씩 총 8개의 Output이 나와야한다. 또한, 우리가 필요한 Output은 2번 index에 존재한다.

### 2-1) Pytorch Model

- Output Length = 8

- Output[2]

```
tensor([[-3.1196, -1.1848,  0.9928, -2.4853]], device='cuda:0')
```

- 성능

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/195778592-9faf93f1-fe81-4823-9606-f517cfa14fe7.png"></p>

<br>

### 2-2) TensorRT Model

- Output Length = 8

- Output[2]

```
tensor([[-3.1207, -1.1842,  0.9931, -2.4848]], device='cuda:0')
```

- 성능

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/195778568-b691e0a1-4b30-47e5-9603-df5fd7e30cbb.png"></p>

<br>

### 2-3) TensorRT Engine

- Output Length = 8

- Output[2]

```
[0. 0. 0. 0.]
```

- 성능

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/195778579-ea1fbbb2-d710-43bc-9b14-2e39070dfd94.png"></p>

<br>
<br>

## 3. 결과

Pytorch -> TensorRT에는 눈부신 성능 향상이 있었지만, TensorRT -> TensorRT Engine에는 별다른 성능 향상은 존재하지 않았다. 또한, TensorRT Engine은 Output값이 이상하게 나왔는데, 해당 모델이 멀티모델이기 때문에 제대로 컨버팅이 되지 않았기 때문이라고 한다. 따라서 그냥 TensorRT 모델을 사용하면 될 것 같다.

<br>
<br>

## 4. 추가목표

현재 ARM 아키텍쳐와 x86에서 사용하는 투기감지 모델이 TensorRT가 아니라 yolov5로 돌고 있다고 한다. 그래서 현장 그래픽카드 온도가 70~80도까지 올라가서 그래픽카드 교체 시기가 짧아지고 있다고 하셨다. 따라서 다음과 같은 작업을 지시받았다.

1. 현장 투기 감지 모델을 TensorRT 모델로 컨버팅하기

2. 현장 투기 감지 코드에 객체/skeleton ssd모델 녹이기

    투기 감지 모델은 객체/skeleton ssd모델의 output값에 영향을 받는다.
    yolov5에도 객체 감지 기능이 있긴 한데 약하므로 ssd 모델을 녹여주자.
