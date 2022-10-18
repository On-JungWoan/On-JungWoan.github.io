---
title:  "투기행위 인식률 개선"
excerpt: "쓰레기 무단투기 탐지 프로젝트"

categories:
  - cctv
tags:
  - [딥러닝, computer_vision, ICT인턴십]

published: true

toc: true
toc_sticky: true
 
date: 2022-09-30
last_modified_at: 2022-09-30
---

## 투기 행위 검출이 잘 안됨
객체 검출은 잘 되는데 투기 행위 인식률이 매우 떨어지는 issue가 발생하였다. 며칠 전 옥상에서 찍은 테스트셋을 모델에 넣어보았는데, 명백히 투기행위로 인식되는 이미지조차 투기 행위로 인식하지 못하고 있었다. 현장에서 사용하는 모델과 다르고, 아직 파인튜닝이 완전히 끝난 모델이 아니긴 했지만, 1500장 당 1~2건밖에 인식을 못할정도로 성능이 너무 안좋아서 차장님께 문의드려서 얻은 결과이다.

<br>
<br>

## 1. Normal Pose에 대한 오버피팅 가능성
클래스별 train 데이터셋의 수가 1:1이 아니라 노멀 포즈에 대한 데이터셋이 압도적으로 많아 해당 클래스에 대해 오버 피팅되었을 가능성이 있다고 하셨다. 현재 모델에서 사용하고 있는 행위 탐지 로직은 다음과 같다.

1. 객체를 검출하여 해당 크기만큼 이미지를 crop한다.
2. crop된 이미지를 CUDA에 넣어 연산하고 다시 호스트에 return한다.
3. return한 output값 중, confidence가 threshold를 넘지 못하는 box는 걸러낸다.
4. 최종적으로 남은 box 중에서 score가 가장 좋은 box를 선택한다.
5. 해당 box의 클래스가 0(투기행위)인지 확인하고, 0이면 투기행위로 인식하여 경고 음성을 출력한다.

하지만 이렇게 될 경우, 노멀 포즈가 오버피팅 되어 있기 때문에 대부분의 case에서 클래스 1(Normal Pose)인 경우가 Max값으로 추출된다. 따라서 투기 행위가 인식될 가능성이 낮아진다는 것이다. 따라서 Max값을 추출할 때, class가 0인것들만 먼저 뽑아서 그 중에서 최대값을 추출하는 방식으로 해당 문제를 해결하였다.

![image](https://user-images.githubusercontent.com/84084372/193232012-7e808d9a-af35-4a16-b073-bd1b883a1a70.png)

<br>
<br>

## 2. threshold 조정

또한, threshold값은 하이퍼 파라미터 느낌이어서 적절하게 값을 조정해 주어야한다고 하셨다. 1차적으로 모든 box가 threshold값을 넘지 못해서 score 자체가 뽑아지지 않는 경우가 발생하였다. 따라서 threshold값을 살짝 낮추어주었다.

<br>
<br>

## 3. 최종 Output

위 방법으로 어느정도 행위 인식률을 개선할 수 있었다(해당 사진은 본인이 직접 옥상에 올라가서 찍은 테스트셋이다).

![image](https://user-images.githubusercontent.com/84084372/193232846-586cb066-01f8-4ad2-980d-2cd2c731b812.png)

<br>
<br>

<div align='center'>
  <a href="https://on-jungwoan.github.io/internship/Itern-TIL/#0930">
    본문으로 돌아가기
  </a>
</div>
