---
title:  "TensorRT Engine Output 가공 / CUDA STREAM 수정"
excerpt: "쓰레기 무단투기 탐지 프로젝트"

categories:
  - cctv
tags:
  - [딥러닝, computer_vision, ICT인턴십]

published: true

toc: true
toc_sticky: true
 
date: 2022-10-17
last_modified_at: 2022-10-17
---

## 1. 진행상황

x86과 ARM 아키텍쳐에서 사용하는 모델 변환 작업을 해주었다. 기존 y_model을 새롭게 변환한 TensorRT 모델로 갈아끼우는 작업을 완료하였다.

<br>
<br>

## 2. TensorRT Engine Output Post-Treatment


TensorRT에서 추출한 engine 파일을 사용하기 위해서 Output값 후처리 작업을 이것저것 해보았다. TensorRT에서 engine 모듈을 추출하는 방법은 다음 포스팅을 참조하면 된다.

- **Ref.**

  [TensorRT 모델에서 engine 모듈 추출하기](https://on-jungwoan.github.io/dl/tensorrtengine/)

원본 Pytorch 모델은 4개의 모델을 Blending한 모델이고, 모델마다 output이 2개씩 존재하여 총 8개의 Output이 나온다. 자세한 내용은 아래 글을 참조하면 된다.

- **Ref.**

  [TensorRT Engine 성능 비교](https://on-jungwoan.github.io/cctv/tensorrt_engine/#2-3-tensorrt-engine)
  
<br>

### 2-1) CUDA Stream 수정

현재 Pytorch -> TensorRT 변환은 완벽하게 완료된 상태이고, TensorRT -> TensorRT Engine 추출과정을 진행중이다. TensorRT에서 engine 모듈만 추출하는 것은 성공하였으나, output의 값이 이상하게 나와서 사용하지 못하고 있는 상태이다. engine에서 뽑힌 output의 형식은 동일한데, 값이 0만 나와서, TensorRT 모델에서 추출한 engine 파일의 output 값을 후처리 해보고자 하였다. 차장님께서는 blending 모델이라 변환이 불가하다고 하셨지만, 잘 변환하면 될 것 같아서 계속 시도해봤다. 


- **기존 방식**

```
input = [
    host : 1 of length
    device : 1 of length
]

output = [
    host : 8 of length
    device : 8 of length
] 
```

```
host_input[0] -> device_input[0] -> 추론 -> device_output[0] -> host_output[0]
```

기존 추론 방식은 host_input[0] 값을 device[0] (cuda)에 넘겨주고, GPU에서 연산을 한 뒤(TensorRT engine인 관계로 NVIDA GPU만 사용가능), 연산 결과를 device_output[0]에 받아서 다시 host_output[0]으로 넘겨주는 형태이다. 여기서 실수가 있었다. 해당 모델의 8개의 output 중, 투기행위에 대한 confidence는 2번 원소에 저장되어있는데, GPU에서 0번 원소만 연산하고 받아와서 다른 Output은 다 0이 나왔던 것이다. 따라서 CUDA Stream 과정을 다음과 같이 수정하였다.

<br>

```
host_input[0] -> device_input[0] -> 추론 -> device_output[2] -> host_output[2]
```

투기행위 추론 과정에 필요한 2번 Output만 CUDA에서 가져왔다. 위 CUDA STREAM에 대한 내용은 다음 글에 적어놓았다.

- **Ref.**

  [알고리즘 흐름 / CUDA Streams](https://on-jungwoan.github.io/cctv/cctv_bounding/#2-3-%ED%95%B4%EB%8B%B9-cuda-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%ED%8C%A8%ED%84%B4)

<br>

### 2-2) Output Post-Treatment

다음은 GPU에서 계산된 output 값이다. 2번 Output만 추출한 것이다.

```
print(output)
```
```
[-5.297592   2.0955591  3.4789176 -5.6720223]
```

<br>

softmax 변환을 해보았다. softmax 인자값으로 dim=X를 사용하라는 warning이 떠서 굳이 차원을 하나 더 추가해줬다.

```
nn.softmax(dim=1)(torch.tensor([output]))[0]
```
```
tensor([1.2335e-04, 0.20043, 0.79936, 8.4828e-05])
```

<br>

4개의 value 중에서 argmax 값이 class의 index가 되고, max값이 해당 클래스에 대한 confidence가 된다.

```
classes = ['Carry', 'Jest Before', 'Dumping', 'Normal']
```

따라서 해당 이미지는 Dumping이고, score는 0.79936이 된다. 기존 TensorRT에서 예측한 결과는 Dumping 0.9935374855995178이다.

### 2-3) TensorRT의 Output과 비교

confidence가 TensorRT의 것과 완전히 같지 않고 편차가 존재해서, input 이미지 위에 TensorRT가 추론한 confidence/class와 TensorRT engine이 추론한 confidence/class를 같이 찍어보았다. TensorRT engine이 추론한 것은 앞에 new를 붙여서 구분했다.

- **Normal Pose**

기존 TensorRT 모델은 해당 모션을 Normal Pose로 인식했으나, TensorRT Engine은 Carry로 인식하였다. 여기서 Carry는, 물건을 들고 움직이는 상태를 의미한다. 오히려 기존 모델보다 더 합당한 추론을 하고 있는 것 같았다.

![image](https://user-images.githubusercontent.com/84084372/196134968-9eee77ca-bec5-4743-a733-1361a5f405c0.png)

<br>

- **Dumping Pose**

Dumping Pose에 대해서는 두 방법 모두 같은 추론을 하고있다. 사실 Dumping Pose가 가장 중요하며, 테스트 대상이므로 Dumping Pose만 잘 나오면 된다.

![image](https://user-images.githubusercontent.com/84084372/196135130-afc5955e-439e-4d9f-b469-85a82fce58b7.png)

<br>
<br>

## 3. 문제점

얼추 맞는 부분도 있지만, 다음과 같이 값이 한번씩 튀는 경우도 종종 생긴다. Output값을 조금 더 후처리해주어야 할 것 같다. TensorRT는 Dumping으로, TensorRT Engine은 Just Before Pose로 추론하였다. Just Before는 단순히 뒷모습에 대한 Pose이다. 원본 Pytorch 모델의 Output과 더 유사한 쪽은 TensorRT이긴 하지만, 어찌보면 engine의 추론이 더 그럴싸해보인다. 어떤 것을 사용할지는 좀 더 많은 테스트셋에 대해서 확인해보고 결정해야 할 것 같다.

![image](https://user-images.githubusercontent.com/84084372/196135314-e345ee68-d779-4e0c-9f74-d8d73f254f91.png)

<br>
<br>

## 4. 추가 목표
차장님께 문의드린 결과, Output을 맞추는 작업은 나중에 하고 일단은 TensorRT와 TensorRT Engine의 성능 차이를 시각화해달라고 하셨다.
일단은 Output값이 얼추 나오는 것을 확인했으니까, 성능을 비교해보기로 하였다. 비교 대상은 일반 Pytorch 모델, TensorRT 모델, TensorRT Engine이다. 연산 속도를 시각화하여 표준편차, 평균 속도 등을 파악해서 차장님께 보고드리기로 하였다.

<br>
<br>

<div align='center'>
  <a href="https://on-jungwoan.github.io/internship/Itern-TIL-10/#1017">
    본문으로 돌아가기
  </a>
</div>
