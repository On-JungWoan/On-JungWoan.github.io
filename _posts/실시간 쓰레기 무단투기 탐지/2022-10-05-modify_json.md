---
title:  "PyQT에 넘겨줄 json 수정"
excerpt: "쓰레기 무단투기 탐지 프로젝트"

categories:
  - cctv
tags:
  - [딥러닝, computer_vision, ICT인턴십, 유클리드소프트]

published: true

toc: true
toc_sticky: true
 
date: 2022-10-05
last_modified_at: 2022-10-05
---

## 1. 기존 객체검출 Confusion Matrix 문제점
기존 방식은, 사진에 사람이 2명 있는데 Bounding Box는 1개밖에 그려지지 않았다면 오분류 한 것으로 판단하였다. 그렇다면 이러한 경우 객체 검출률에 대한 Confusion Matrix는 어떻게 그려야 할까? FP일까 FN일까? Confusion Matrix를 그리기가 상당히 애매해진다. 따라서 각 개체마다의 라벨링 데이터와 모델에서 예측한 box와 간의 유사도가 50%이상이면 잘 검출한 것으로 판단하기로 하였다.

<br>
<br>

## 2. 새로운 방식의 json
위 문제점을 해결하기 위해서는 기존에 PyQT로 넘겨주던 json의 데이터를 바꿀 필요가 있었다. 아래는 같은 사진에 대한 기존 json 표현 방식과 새로운 json 표현 방식이다. 총 3개의 객체가 검출된 경우인데, 새로운 json의 경우 각각 객체에 대하여 Bounding Box의 좌표정보(x1,y1,x2,y2)와 투기 여부가 따로 출력되고 있다. 기존 json의 경우, 사진에서 투기행위가 1건이라도 감지되는 경우 무조건 'Y'를 리턴한다. 

<div align="center">
    <strong>[새로운 json]</strong>
</div>

```json
{"people":
    [
        {
            "box": [674, 194, 870, 503],
            "dumping_yn": "Y"
        },
        {
            "box": [674, 194, 870, 503],
            "dumping_yn": "N"
        },
        {
            "box": [674, 194, 870, 503],
            "dumping_yn": "N"
        }
    ]
}
```

<div align="center">
    <strong>[기존 json]</strong>
</div>

```json
{
    "dumping_yn": [
        "Y"
    ],
    "people": [
        3
    ]
}
```

<br>
<br>

## 3. 로직 수정

최종 return값은 다음과 같이 변경된다. 객체의 투기 여부를 저장하는 리스트인 dumping_list와 Bounding Box의 좌표를 저장하는 box_list를 새롭게 만들어 주었다. 객체의 수와 전체 투기여부는 해당 내용으로 대체가 가능하므로 제거해주었다.

```python
return dumping_list, box_list, image, box_image
```

- **알고리즘 흐름**

![image](https://user-images.githubusercontent.com/84084372/194017674-7a4d81a9-cf41-4831-b26b-37fd07a25888.png)

### 3-1) 각 개체에 대한 Bounding Box 좌표정보 List 추가

첫 번째 SSD Model에서 출력된 Box의 수(객체 수)만큼 for문이 돌고 해당 for문 안에서 Box 좌표와, 투기 여부가 계산된다. 
Box의 좌표는 이미지를 Crop하는 과정에서 계산되므로, 해당 코드 밑에 box_list.append하는 코드를 작성하였다. 

<br>

### 3-2) 각 개체에 대한 Dumping 여부 List 추가

일단 dumping_yn에 default 값으로 'N'을 지정하고, CUDA에서 return된 output 값 중 Treshold를 넘는 case가 발생하면 'Y'를 return하였다. 
for문 끝단에 dumping_list에 append하는 코드를 추가해주었다.


<br>
<br>

<div align='center'>
  <a href="https://on-jungwoan.github.io/internship/Itern-TIL-10/#1005">
    본문으로 돌아가기
  </a>
</div>
