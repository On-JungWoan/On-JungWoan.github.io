---
title:  "Convert DETR-like models to TFLite"
excerpt: "DETR-like models for edge device"

categories:
  - model-optimization
tags:
  - [DL, computer_vision]

published: true

toc: true
toc_sticky: true
 
date: 2023-04-18
last_modified_at: 2023-04-18
use_math: true
permalink: "/dl_optim/ddetr_to_tflite/"
legacy_category: DL_optim
---

![image](https://user-images.githubusercontent.com/84084372/232539845-4c3c2d56-4c80-4fff-8037-31ced3fedaa3.png)

최근 발표된 SOTA 모델의 경우, 대부분 그 크기가 매우 크고 무거움(특히 비전분야). 이러한 pre-trained 모델들을 개발 환경에서 사용할 때는 대부분 큰 문제가 되지는 않는다. 하지만 Android나 임베디드 보드와 같은 엣지 디바이스에서는 이러한 Full-size 모델을 사용하는 데 한계가 존재함. 따라서 torch 모델을 엣지 디바이스에서 사용하기 위해서는, 이를 최적화 된 포맷으로 변환해주는 작업이 필요함. 해당 포스팅에서 사용할 torch 모델은 요즘 핫한 Transformer based의 DETR계열 모델이며, 이 모델을 엣지 디바이스에 최적화가 아주 잘 되어있는 `TFLite`로 변환하는 과정에 대해서 소개하도록 하겠음.

<br>
<br>

# 1. Torch to ONNX

## 1-1. ONNX: Open Neural Network Exchange

Torch를 TFLite로 변환하기 위해서 Torch를 ONNX라는 중간 포맷으로 변환하고, 변환된 ONNX를 다시 TFLite로 변환하는 과정을 거치게 됨. 여기서, ONNX란 Open Neural Network Exchange의 줄임말로써, 서로 다른 DNN 플랫폼에서 작성된 모델을 호환하여 사용할 수 있도록 도와주는 공유 플랫폼임.

<br>

## 1-2. 변환 과정

![image](https://github.com/On-JungWoan/On-jungWoan/assets/84084372/865f0fd9-9e7c-4e89-a0c6-2bb697d18aa3)

먼저, torch를 onnx로 변환하기 위해서 torch.onnx.export() 메서드를 사용함. 해당 메서드는 model과 dummy input을 인자로 받아, model에 dummy input을 흘려 보내줌. 입력된 dummy input은 모델을 타고 흐르면서 1차적으로 ONNX IR(Intermediate Representation)을 생성함. 그리고 여기서 한번 더 ONNX IR에 대해 graph optimization을 거친 뒤 최종적으로 ONNX 그래프 형태로 model을 저장하게 됨.

<br>

## 1-3. 사용 버전

변환 시 사용한 ONNX의 버전은 1.11.0이며, 오프셋 버전은 16을 사용함.

| ONNX version | IR version | Opset version ai.onnx | Opset version ai.onnx.ml | Opset version ai.onnx.training
| :-: | :-: | :-: | :-: | :-: |
1.11.0 | 8 | 16 | 3 | 1

<br>
<br>

# 2. ONNX to TFLite

## 2-1. 변환 라이브러리 : onnx2tf

변환된 ONNX 모델을 TFLite로 변환하기 위해서, ONNX와 TFLite의 데이터 포맷을 맞추어주는 작업이 필요함. 변환된 ONNX의 데이터 포맷은 NCHW 형식인 반면, TFLite에서 지원하는 데이터 포맷은 NHWC 형식임. 이러한 포맷의 차이로 인해 ONNX에서는 채널로 인식되던 차원이 TFLite에서는 H로 인식되는 등 포맷 문제가 발생할 수 있음. 따라서 레이어마다 데이터 포맷이 맞지 않는 부분이 있는지 확인하고, 맞지 않는 부분은 직접 포맷을 변경해주어야 하는 작업이 필요함. 해당 작업은 많은 시간을 필요로 하기 때문에, 이러한 포맷 차이를 해결하는 데 특화되어 있는 onnx2tf라는 변환 라이브러리를 사용.

<br>

## 2-2. 모델 최적화 : ONNX simplifier

![image](https://github.com/On-JungWoan/On-jungWoan/assets/84084372/5aa034a3-3982-4728-ad0b-30896ab38ec1)

onnx2tf의 모델 변환 과정은 크게 모델 최적화, 연산자 변환, 포맷 변환으로 나누어짐. 먼저 ONNX 모델이 input으로 들어오게 되면, 해당 모델을 ONNX simplifier를 사용하여 변환에 편리한 형태로 최적화해 줌. 이 때, 최적화 과정에는 불필요한 연산 제거 및 constant folding 등이 포함됨. 이를 통해 불필요한 연산자의 변환을 생략함으로써 변환 속도를 향상시키고, 변환과정에서 발생할 수 있는 오류의 발생확률을 줄일 수 있음.

<br>

## 2-3. 연산자 변환

모델의 최적화가 끝나면, ONNX 연산자를 TFLite 연산자로 변경해 주는 과정을 거침. 각 ONNX 연산자마다 대응하는 TFLite 연산자를 미리 정의해두고, 모델의 모든 레이어를 순회하며 미리 정의해 둔 대응관계에 맞게 연산자를 치환함. 예를 들어, ONNX의 Add 연산자는 TFLite의 tf.add로 변환되도록 변환 관계를 정의하였다면, 해당 관계에 의해 모든 Add 연산자는 tf.add 연산자로 치환됨. 그러나 위와 같이 일대일 대응이 되는 TFLite 연산자가 존재하지 않을 경우 치환할 연산자 외에도 추가적인 변환과정을 같이 정의해야 함. 가령, 나누기 연산의 경우 ONNX에서는 지원하는 연산자이지만, TFLite에서는 지원하지 않는 연산자임. 이 경우, 나누는 수에 역수를 취해준 뒤 나눗셈 연산을 곱셈 연산으로 바꾸어주는 작업이 필요함. 최종적으로, 해당 과정은 ONNX의 나눗셈 연산과 일대일 대응되어 치환됨.

<br>

## 2-4. 포맷 변환

`2-1.`에서 언급한 바와 같이, ONNX(pytorch)와 TFLite의 입력 포맷이 서로 달라 연산자가 입력 차원을 오인식하는 경우가 발생할 수 있음. 이러한 차원의 불일치를 해결하기 위해 모든 연산자에 대해서 일치 여부를 확인해줘야 하지만, onnx2tf는 불일치가 일어나는 연산자를 자동으로 찾아서 출력해주는 기능을 제공함. 따라서 출력된 연산자의 입력 차원에 대해서 transpose를 통해 TFLite에 맞는 입력 포맷으로 변경해 줌.

<br>

## 2-5. 검증

![image](https://github.com/On-JungWoan/On-jungWoan/assets/84084372/4b1827e9-8a7b-4757-b6c5-f3b9f0e28cf8)

ONNX의 모든 연산자를 TFLite 연산자로 치환한 뒤, 각 연산자에 대한 검증 과정을 수행함. 검증을 위해 모델의 입력 크기와 동일한 더미 데이터를 생성하고, 변환된 TFLite 모델과 원본 ONNX 모델에 각각 넣어줌. 이후 치환된 연산자와 기존 연산자간의 오차를 측정함. 이 때, 측정된 오차가 미리 지정한 범위 안에 포함되면 연산자의 변환이 완료되며, 그렇지 않은 경우 해당 연산자부터 변환을 다시 시작함. 오차는 np.allclose()를 사용하여 측정되며, 허용 오차의 결정식은 다음과 같음. 이 때, atol과 rtol은 사용자가 지정함.

$$
|a-b| <= ( atol + rtol * |b| )
$$

<br>
<br>

# 3. TFLite

## 3-1. Deformable DETR & DN DETR

![image](https://github.com/On-JungWoan/On-jungWoan/assets/84084372/71d0c36d-f4ff-4a22-98ba-455e1dfcc066)

본인이 TFLite로 변환에 성공한 모델에는 Deformable DETR과 DN DETR이 있음. Deformable DETR의 경우, TFLite로 변환시 발생하는 현저한 추론 속도 저하가 있음. 속도 저하의 원인은 Deformable attention 과정에서 사용되는 Grid Sample 연산자인데, 해당 연산자는 일대일 대응이 되는 TFLite의 연산자가 없음. 따라서 `2-3`에서 언급한 바와 같이 일련의 추가적인 변환과정을 거쳐야 하는데, 이 과정에서 추론 속도 저하가 발생함. 따라서 실제 inference시에는 Deformable attention을 사용하지 않는 DN-DETR을 사용하는 게 좋음.

<br>

## 3-2. Inference time

![image](https://github.com/On-JungWoan/On-jungWoan/assets/84084372/b2961f66-b99e-40f6-a6f8-73c2c18ec78d)

파란색 그래프는 cpu 환경에서 측정한 이미지 한 장당 추론시간이며, 빨간색 그래프는 TFLite에서 측정한 추론시간 그래프임. x축은 각각 이미지 크기를, y축은 초단위 추론시간을 의미함. Deformable DETR(좌측 figure)의 경우 Grid sample 연산자의 영향으로 인해, TFLite로 변환 시 급격한 추론 속도의 저하를 보임. 반면 DN-DETR(우측 figure)의 경우 TFLite로 변환하여도 속도 저하가 거의 나타나지 않고 있음.

![image](https://github.com/On-JungWoan/On-jungWoan/assets/84084372/29aea33b-eb7a-435e-ac4e-925b40c5e3df)

TFLite에서 이미지 한 장을 추론하는 데 Deformable DETR은 평균 28.6초가 소요되었으며, DN DETR은 평균 1.5초가 소요됨.

<br>

## 3-3. Input size

![image](https://github.com/On-JungWoan/On-jungWoan/assets/84084372/f5e11e27-e659-4dab-9113-1f4fca39bce4)

파란색 막대 그래프는 TFLite에서 측정한 이미지 한 장당 추론 시간을 의미하며, 빨간색 선 그래프는 mAP를 의미함. x축은 input size를 나타내며, 좌측 y축은 추론 시간, 우측 y축은 mAP를 나타냄. 위 그래프를 통해 input size(or 추론시간)와 mAP간에는 Trade-Off 관계가 존재함을 알 수 있음. 또한, Torch와 TFLite의 mAP가 매우 작은 오차범위(1e-4) 내에서 일치하고 있는 것을 확인함. 이를 통해 변환이 성공적으로 수행되었음을 확인할 수 있음.