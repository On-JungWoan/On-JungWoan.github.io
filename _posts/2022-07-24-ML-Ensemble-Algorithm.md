---
title:  "[ML 이론]앙상블(Ensemble) 알고리즘"
excerpt: "배깅(Bagging)과 부스팅(Boosting)을 중심적으로"

categories:
  - Blog
tags:
  - [jekyll, Github, Githubio]

published: true

toc: true
toc_sticky: true
 
date: 2022-07-24
last_modified_at: 2022-07-24
---

<br>

머신러닝 앙상블이란 **여러개의 머신러닝 모델을 이용해** 최적의 답을 찾아내는 기법.   
정형 데이터 분류 및 예측 시 뛰어난 성능을 나타내는 기법이다.   
앙상블 기법의 종류는 다음과 같다.   

- 보팅(Voting)
  - 여러 알고리즘 모델을 조합하여 투표를 통해 결과 도출
- 배깅(Bagging)
  - Low Variance, High Bias
  - 하나의 알고리즘 내에서 샘플 중복 생성을 통해 결과 도출
- 부스팅(Boosting)
  - High Variance, Low Bias
  - 이전 오차를 보완하며 가중치 부여
- 스태킹(Stacking) : 이 포스팅에서는 다루지 않겠다.

<br>

<div align="center">
    <img src="https://user-images.githubusercontent.com/84084372/180629232-841f953c-141f-48b8-b69b-1a046e9cf017.png">

    출처:http://scott.fortmann-roe.com
</div>

<br>

## 1. 보팅 (Voting)
투표를 통해 결과를 도출하는 방식이다. 투표를 하여 결과를 도출한다는 점에서 배깅과 비슷하지만, 보팅은 여러 알고리즘 모델을 사용한다는 점에서 배깅과 차이가 있다. 
분류 문제에서 보팅 알고리즘의 종류는 다음과 같다.

<br>

### 1.1 Hard Voting
- 다수의 분류기가 결정한 예측값을 선택

<div align="center">
    <img src="https://user-images.githubusercontent.com/84084372/180629483-dcdb838b-1ed6-4ffc-84bb-3ed7e91a3f7e.png">

    출처:https://velog.io/@jiselectric/Ensemble-Learning-Voting-and-Bagging-at6219ae
</div>

<br>

### 2.1 Soft Voting
- 결정 확률의 평균값을 구한 뒤, 확률이 가장 높은 레이블값을 최종 선택
- 일반적으로 많이 사용한다.

<div align="center">
    <img src="https://user-images.githubusercontent.com/84084372/180629511-c1dda79a-c8f5-4ac8-afd3-e9ae4d10d35b.png">

    출처:https://velog.io/@jiselectric/Ensemble-Learning-Voting-and-Bagging-at6219ae
</div>

<br>

## 2. 배깅 (Bagging)
Bagging은 Bootstrap(샘플) + Aggregating(합산)의 줄임말로, 한 알고리즘 내에서, 여러개의 dataset 중첩을 허용하게 하여 샘플링하는 방식이다. 
사용성이 쉽고, 앙상블 알고리즘 중 비교적 빠른 수행속도를 가지고 있다. 
초기 모델링 시, 배깅으로 모델의 성능을 어느정도 가늠해본 뒤, 부스팅 알고리즘을 사용하여 성능을 높이는 방식을 채택한다. 

<br>

### 2.1 Random Forest

- 특징
  - DecisionTree 기반 Bagging 앙상블
  - 사용성이 쉽고 성능이 우수하여 초기 모델링에 많이 쓰임
  - 과정 : 배깅 -> 무작위 변수 선택 -> 변수 중요도 산출

<br>

- 주요 Hyper Parameter
    - random_state: 랜덤 시드 고정 값. 고정해두고 튜닝할 것!
    - n_jobs: CPU 사용 갯수
    - max_depth: 깊어질 수 있는 최대 깊이. 과대적합 방지용
    - n_estimators: 앙상블하는 트리의 갯수
    - max_features: 최대로 사용할 feature의 갯수. 과대적합 방지용
    - min_samples_split: 트리가 분할할 때 최소 샘플의 갯수. default=2. 과대적합 방지용

<br>

## 3. 부스팅 (Boosting)
약한 학습기를 순차적으로 학습 하되, 이전 학습에 대하여 잘못 예측된 데이터에 가중치를 부여함으로써 오차를 보완. 
순차적 학습을 하여 오차를 보완하는 방식이기 때문에 배깅과 달리, 병렬처리에 어려움이 있음. 
따라서 부스팅 알고리즘은 성능이 매우 우수하다는 장점이 있지만, 다른 앙상블 대비 학습 시간이 오래 걸린다는 단점이 존재한다. 
또한, Light GBM의 경우, 과대적합(Overfitting)에 취약하다.

<br>

### 3.1 GradientBoost, XGBoost

- 특징
    - 성능이 우수한 Boosting 알고리즘이다.
    - 하지만 학습시간이 매우 느리다는 단점이 있다.
    - XGBoost가 GradientBoost보다는 빠르고 성능도 향상되긴 하였으나 여전히 매우 느린 학습시간.
    - XGBoost는 Scikit-Learn 패키지가 아니다.

<br>

- 주요 Hyper Parameter
  - random_state: 랜덤 시드 고정 값. 고정해두고 튜닝할 것!
  - n_jobs: CPU 사용 갯수
  - learning_rate: 학습율. 너무 큰 학습율은 성능을 떨어뜨리고, 너무 작은 학습율은 학습이 느리다. 적절한 값을 찾아야함. n_estimators와 같이 튜닝. default=0.1
  - n_estimators: 부스팅 스테이지 수. (랜덤포레스트 트리의 갯수 설정과 비슷한 개념). default=100
  - subsample: 샘플 사용 비율 (max_features와 비슷한 개념). 과대적합 방지용
  - min_samples_split: 노드 분할시 최소 샘플의 갯수. default=2. 과대적합 방지용

<br>


### 3.2 Light GBM

- 특징
    - 성능이 우수한 Boosting 알고리즘이다.
    - Boosting 계열 알고리즘이 가지는 치명적 단점인 느린 학습 속도를 개선
    - Leaf-wise(DFS와 같이 트리를 만들어 나감)
    - scikit-learn 패키지가 아님 (Microsoft 사 개발)

<br>

- 주요 Hyper Parameter
  - random_state: 랜덤 시드 고정 값. 고정해두고 튜닝할 것!
  - n_jobs: CPU 사용 갯수
  - learning_rate: 학습율. 너무 큰 학습율은 성능을 떨어뜨리고, 너무 작은 학습율은 학습이 느리다. 적절한 값을 찾아야함. n_estimators와 같이 튜닝. default=0.1
  - n_estimators: 부스팅 스테이지 수. (랜덤포레스트 트리의 갯수 설정과 비슷한 개념). default=100
  - max_depth: 트리의 깊이. 과대적합 방지용. default=3.
  - colsample_bytree: 샘플 사용 비율 (max_features와 비슷한 개념). 과대적합 방지용. default=1.0

<br>

### 3.3 CatBoost

- 특징
    - 범주형 변수로 이루어진 데이터셋에 특화되어 있는 부스팅 알고리즘이다.
    - XGBoost, LightGBM보다 빠른 학습 속도
    - Hyper Parameter에 따라 성능이 달라지는 문제 해결
    - 부스팅 알고리즘 특성상, Bias-Variance Tradeoff에 의해 Variance가 높아 오버피팅이 발생하기 쉽지만, CatBoost는 이를 해결
    - Level-wise(BFS와 같이 트리를 만들어 나감)
    - Orderd Boosting 방식을 사용
    - 즉, 순서(order)에 따라 모델을 학습시키고 순차적으로 잔차를 계산하는 과정을 반복
    - 결측치가 매우 많은 데이터셋에는 부적합한 모델이며, 수치형 데이터에서는 LightGBM보다 학습 속도가 떨어짐

<br>

- 주요 Hyper Parameter
    > 기본 파라미터가 기본적으로 최적화가 잘 되어있어서, 파라미터 튜닝에 크게 신경쓰지 않아도 된다. (반면 xgboost 나 light gbm 등의 부스팅 기법은 파라미터 튜닝에 매우 민감하다.) Catboost는 이를 내부적인 알고리즘으로 해결하고 있어서, 파라미터 튜닝할 필요가 거의 없다. 굳이 한다면 learning_rate, random_strength, L2_regulariser 과 같은 파라미터 튜닝인데, 결과는 큰 차이가 없다고 한다.
  - has_time : 시간이 지남에 따라 데이터가 변경되는 경우 True 로 설정하고 수행 가능하다.
  - fold_len_multiplier(must be>1), approx_on_full_history : 데이터가 적은 경우 각각 1과, true로 설정한다. 데이터가 적은 경우, Catboost는 각각의 데이터에 다른 모델을 이용해서 잔차를 계산할 수 있다.
  - task_type : 대규모 데이터셋일 경우, GPU 로 설정한다.

<br>
