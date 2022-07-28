---
title:  "[딥러닝] 작물 잎 사진으로 질병 분류하기 예제 풀이 #2"
excerpt: "교재 : 파이토치 딥러닝 프로젝트 문제집"

categories:
  - DL
tags:
  - [딥러닝, CNN, 전이학습]

published: true

toc: true
toc_sticky: true
 
date: 2022-07-28
last_modified_at: 2022-07-28
---

## 전이학습
ImageNet 훈련 데이터를 이용하여 Pre-Trained된 Model을 사용한다. 
해당 파라미터 값 일부를 우리가 가진 데이터셋의 특성에 맞게 조정하며, 이를 Fine-Tuning 이라고 한다. 
여기서 유의해야 할 점은, 모든 파라미터를 업데이트 하진 않는다는 것이다. 
다른 종류의 이미지라도 낮은 수준의 특징은 상대적으로 비슷할 가능성이 높기 때문이다. 
따라서 분류기와 가까운 Layer부터 원하는 만큼 학습 과정에서 업데이트하며, 나머지 Layer는 Freeze한다. 
이 때, Freeze Layer 수는 pre-trained model에 사용된 데이터셋과의 유사성을 고려하여 결정한다. 
다음 사진은, Freeze Layer의 수를 결정하는 전략이다. 

![image](https://user-images.githubusercontent.com/84084372/181512540-12cc7327-f854-4ba3-8a54-c7990caed605.png)

- 데이터 수가 많고, 이용된 데이터와의 유사도 높음 (Quadrant2)
    - 분류기와 가까운 일부 Layer만을 훈련시키고, 나머지는 Freeze 한다.
    - 데이터의 유사도가 높기 때문에 상대적으로 적은 수의 Layer만 업데이트 해도 성능이 좋음

- 데이터 수가 적고, 이용된 데이터와의 유사도 높음 (Quadrant4)
    - Convolutional Base 부분 전체를 Freeze하고, 분류기 부분만 변경한다.
    - 일부 Layer를 재학습하기에는 오버피팅의 위험 높음.

- 데이터 수가 많고, 이용된 데이터와의 유사도 낮음 (Quadrant1)
    - 전체 데이터를 Unfreeze하여 학습을 진행
    - 모든 Layer가 학습 과정에서 업데이트 되지만, pre-trained 모델의 효과적인 구조는 그대로 유지 됨.

- 데이터 수가 적고, 이용된 데이터와의 유사도 적음 (Quadrant3)
    - 일부 Layer를 freeze하고, 나머지는 전부 학습한다.
    - 모든 Layer를 전부 Unfreeze하기에는, 데이터 크기가 작기 때문에 오버피팅의 위험이 있다.
