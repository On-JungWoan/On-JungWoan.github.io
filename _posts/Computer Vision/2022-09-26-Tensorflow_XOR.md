---
title:  "Tensorflow로 XOR문제 해결"
excerpt: "모두를 위한 딥러닝 강좌 시즌 1"

categories:
  - DL_study
tags:
  - [DL, Tensorflow, ICT인턴십, 모두를 위한 딥러닝 강좌 시즌 1, computer_vision]

published: true

toc: true
toc_sticky: true
 
date: 2022-09-26
last_modified_at: 2022-09-26
til: 'true'
---

## XOR 문제?
과거 딥러닝 모델은 하나의 퍼셉트론만을 사용한 단층 퍼셉트론 모델이었다. 
하지만 이러한 하나의 퍼셉트론만으로는 XOR 문제를 해결하는 것은 불가능하다. 
![image](https://user-images.githubusercontent.com/84084372/192198584-0240d021-3673-43f5-b60b-a4b2b72a62ae.png)

다음과 같은 여러개의 결정 경계가 필요한데, 그러기 위해서는 3개의 퍼셉트론을 사용한 다층 퍼셉트론 모델을 사용해야한다. 

![image](https://user-images.githubusercontent.com/84084372/192198642-a5984634-90b9-4c03-940b-28b5e5fb834e.png)

<br>

## 다층 퍼셉트론

다음과 같은 가중치와 bias를 갖는 3개의 퍼셉트론을 사용하면 XOR문제를 해결할 수 있다.

![image](https://user-images.githubusercontent.com/84084372/192197471-d59af995-057d-471f-b0cb-33c985e21ff1.png)

![image](https://user-images.githubusercontent.com/84084372/192197784-c7e03e2c-ddde-4820-acbe-03c1d112e056.png)

수식으로 나타내면 다음과 같으며, Tensorflow를 사용하여 아래와 같이 표현할 수 있다.


![image](https://user-images.githubusercontent.com/84084372/192199784-c4e9aa92-18be-4756-aa57-03084fe71703.png)

```python
K = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(K, W2) + b2)
```

<br>

## 가중치 설정 문제
하지만 이러한 다층 퍼셉트론 모델은 가중치 설정이 어렵다는 문제점이 있었다. 
해당 문제를 해결하기 위해 은닉층이 도입 되었다.