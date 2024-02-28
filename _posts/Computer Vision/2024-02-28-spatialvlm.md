---
title:  "[CVPR 2024] SpatialVLM 논문 리뷰"
excerpt: "Spatial VLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities"

categories:
  - DL_paper
tags:
  - [DL, computer_vision]

published: true

toc: true
toc_sticky: true
 
date: 2024-02-28
last_modified_at: 2024-02-28
use_math: true
---

> 논문링크 : <a href="https://arxiv.org/pdf/2401.12168.pdf" target="blank_">Spatial VLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities</a>

> Implementation : <a href="https://github.com/remyxai/VQASynth" target="blank_">https://github.com/remyxai/VQASynth</a>

오늘 소개드릴 페이퍼는 `Spatial VLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities`입니다. 구글 딥마인드에서 2024년 1월에 publish하였으며, CVPR 2024에 accept 되었습니다.

<br>
<br>

## 1. Abstract & Introduction

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/717fffd8-0c13-4b1f-b09b-40299272112a" style="border: solid black 1px"></p>

&nbsp;&nbsp;&nbsp;&nbsp;VQA와 robotics 분야에 있어서 spatial relationship에 대해 이해하는 것은 상당히 중요합니다. CLIP등으로 대표되는 VLM이 최근 VQA 벤치마크에서 주목할만한 성능을 보였으나, 여전히 3D space에 대한 이해능력은 많이 뒤떨어지는 실정입니다. 위 figure에서 언급된 바와 같이, 모델은 input image에서 baseball player와 black man 사이의 거리를 인지하지 못합니다. 이러한 spatial information을 이해하는 능력은 그 자체로도 매우 쓸모있을 뿐만 아니라, robotics나 AR등의 downstream task에서도 유용하게 사용될 수 있습니다. 따라서 이러한 limitation을 해결하는 것이 VLM에 있어서 가장 중요한 도전과제라고 할 수 있습니다.

&nbsp;&nbsp;&nbsp;&nbsp;저자는 이러한 한계의 원인이 3D spatial knowledge의 부족에 있다고 하였으며, 이러한 문제를 인터넷에서 수집된 데이터만을 사용하여 해결하려는 것이 문제라고 밝히고 있습니다. 따라서 저자는, 이러한 문제를 해결하기 위해 `3D spatial VQA 데이터를 automatic하게 generation`하는 framework를 개발하였으며, 1000만개의 real-world image로부터 20억개의 VQA example을 생성하였다고 합니다. 이러한 데이터셋을 사용하여 VLM을 훈련함으로써, spatial VQA에 대한 질적/양적인 성능을 상당히 끌어올릴 수 있었으며, spatial reasoning과 robotics 분야로의 새로운 downstream applications을 가능케하였다고 합니다.

<br>
<br>

## 2. Method

### 2.1. Spatial Grounding from 2D Images

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/5b9c1b36-d838-45ae-9c5f-a5ec4b6d6e88" style="border: solid black 1px"></p>

&nbsp;&nbsp;&nbsp;&nbsp;우선, 저자들은 최근 VLM의 공간 추론능력에 대한 한계점이 모델의 구조 때문이 아니라, spatial reasoning에 대한 학습 데이터가 부족했기 때문일 거라고 추측하고 있습니다. 따라서, `spatial reasoning에 대한 질의를 포함하고 있는 VQA data를 생성`하는 것이 해당 연구에 있어서 가장 중요한 부분입니다. 이에 대한 전체적인 과정이 위 figure에 간략히 표현되어 있습니다. 이제부터는 figure의 각 단계에 대해 조금 더 자세히 설명드려보도록 하겠습니다.

<br>

- **(a) Semantic Filtering**

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/d09651cb-1649-43df-8f05-b02b3252a26f" style="border: solid black 1px"></p>

&nbsp;&nbsp;&nbsp;&nbsp;generation의 첫 단계는, `필요 없는 이미지를 필터링`하는 과정으로부터 시작됩니다. 인터넷에서 수집된 데이터셋은 VLM의 학습에 광범위하게 사용되지만, spatial한 정보를 학습하기 위해서는 이러한 데이터셋을 바로 학습할 수 없습니다. 인터넷에 돌아다니는 이미지의 대부분은 배경이 없거나, 딱 하나의 object만을 포함하고 있는 경우가 많습니다. 실제로 구글에 ketchup에 대한 이미지를 검색했을 때, 위와 같이 background가 없거나 single object에 대한 이미지인 경우가 대다수입니다.

&nbsp;&nbsp;&nbsp;&nbsp;이러한 이미지에서는 어떠한 spatial한 단서를 얻을 수 없기 떄문에(인간조차도 이러한 이미지에 대해서는 어떠한 단서도 얻을 수 없음), 우선적으로 해당 이미지는 학습 데이터셋에서 제외하는 과정을 거칩니다. 이를 위해 CLIP base의 open-vocabulary classification model을 사용함으로써 모든 이미지를 분류하고, 필요없는 이미지는 필터링합니다.

<br>

- **(b) Object-centric Contexts Extraction from 2D Images**



&nbsp;&nbsp;&nbsp;&nbsp;다음으로는, Question에 쓰일 text를 만들기 위해 `Region Captioning`을 수행합니다. 이를 위해, 먼저 Region Proposal Network(RPN)을 사용하여 bounding box를 찾아내고, Non-Maximum Suppression(NMS)를 통해 하나의 object에 대해 여러개의 bounding box가 detection되지 않도록 합니다. 그 이후 Image, BBox, Caption Length를 [FlexCap(Anonymous, 2024)](https://openreview.net/pdf?id=7Phicg0WAg)의 input으로 넣어 BBox에 대한 Caption을 생성해냅니다. 여기서, FlexCap은 ICLR2024에 제출된 페이퍼이며, input bbox에 대한 caption을 length만큼의 word를 사용하여 표현해줍니다. SpatialVLM은 Length로 1~6사이의 random한 값을 사용했다고 합니다. 해당 caption은 추후 question template에 사용됩니다.


<br>

- **(d) Ambiguity Resolution**

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/02aff9a2-f8a3-4a5c-afcf-4bee64cfb420" style="border: solid black 1px"></p>


&nbsp;&nbsp;&nbsp;&nbsp;종종 하나의 이미지 안에 비슷한 카테고리를 가지는 object가 여러개 있을 수 있는데, 이 경우 비슷한 caption이 여러개 생성되므로 추후 Question 생성 시 ambiguity를 유발할 수 있다고 합니다(가령, "cake"라는 caption이 한 이미지 안에 여러개 있다면, 해당 caption이 여러개의 object를 가리키게 됨). 이를 해결하기 위해, 저자들은 다음과 같이 2개의 approach에 대해 고려하였습니다.

  1. 일반적인 object detector의 사용 지양
     - 이러한 detector를 사용할 경우, 단순한 카테고리(ex. cake)만을 produce하는 경향이 있다고 합니다.
     - 이것이 바로 FlexCap을 Captioning Model로 채택한 이유입니다(caption이 1~6 사이의 random한 length 가짐).
  2. 추가적인 post-processing algorithm 구현
     - 이외에도 Ambiguity를 해결하기 위해 추가적인 augment, remove과정을 거칩니다.
     - 이를 위해, CLIP을 사용하여 similarity를 계산하고, 이 similarity가 특정 threshold 이상이면 유사한 caption으로 판단합니다.
     - 만약 유사한 caption이,
       - `정확히 2개 존재하는 그룹`이라면, 'to the left'와 같은 spatial attribute를 caption에 추가해주는 augment 과정을 거칩니다.
       - `2개보다 많이 존재하는 그룹`이라면, ambiguity를 제거하기 위해 해당 caption은 delete합니다.

<br>

- **(c) Lifting 2D Contexts to 3D Contexts**

&nbsp;&nbsp;&nbsp;&nbsp;지금까지는 Question을 생성하기 위한 일련의 과정이었으며, 이제부터는 Answer를 생성하기 위한 과정입니다. 기존 object detection이나 bounding box positioning을 사용하여 얻어진 VQA dataset은 2D(and pixel-level)의 reasoning 수준에 머물러 있습니다. 그 이유는 depth, spatial information이 포함된 context가 부족하기 떄문입니다. 따라서 저자들은 이러한 문제점을 해결하기 위해, 우선 `2D pixel을 metric-scale의 3D point cloud로 lift` 해주었습니다.

&nbsp;&nbsp;&nbsp;&nbsp;이를 위해, 먼저 `Depth Estimation`을 진행해줍니다. Depth Estimation에는 SOTA 모델인 ZoeDepth를 사용하였으며, 이를 통해 2D image를 3D point cloud로 lift합니다. 또한, DBSCAN이라는 클러스터링 알고리즘을 사용하여 outlier들을 제거하였다고 합니다.

&nbsp;&nbsp;&nbsp;&nbsp;이후에는 point cloud를 `camera coordinate에서 world coordinate으로 transform`해줍니다. 이를 위해 작은 segmentation model을 학습시켜서 floor나 table top등의 horizontal surface를 찾아내고, 이를 z축으로 설정한 뒤 camera의 original z-axis를 새로운 x축으로 설정함으로써 coordinate canonicalization을 진행한다고 합니다. 만약 이미지에서 horizontal surface가 감지되지 않는 경우에는, elevation과 같이 canonicalization에 대한 의존성이 높은 question을 사용하지 않았다고 합니다. 자세한 내용은 아래 pseudo code를 참고해주세요.

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/1a2e07ba-9a73-4154-b050-86f4992cbb0c" style="border: solid black 1px"></p>

<br>

- **(e) Large-Scale Spatial Reasoning VQA Dataset**

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/77d189af-adcb-4420-b303-e2e93829a7e7" style="border: solid black 1px"></p>

&nbsp;&nbsp;&nbsp;&nbsp;이제 Q&A를 생성하기 위한 모든 준비과정이 끝이 났습니다. Question은 (d) 과정을 거친 caption으로부터 생성하며, 이에 대한 Answer는 (c)의 point cloud로부터 얻습니다.

&nbsp;&nbsp;&nbsp;&nbsp;Question의 경우, 2개 이하의 object에 대한 정보를 포함하며, 크게 Quanlitative question과 Quantitative questions으로 나누어집니다. `Qualitative questions`은 “Given two objects A and B, which is more towards the left?”와 같이 spatial relation에 대한 판단을 요구하는 질문입니다. `Quantitative questions`은 “how much to the left is object A compared to object B?”와 같이 구체적인 수치를 요구하는 질문입니다. Answer의 경우는, 앞서 뽑아낸 point cloud와 3D bounding box를 저자가 개발한 function에 input으로 넣어줌으로써 생성된다고 합니다.

&nbsp;&nbsp;&nbsp;&nbsp;저자는 총 38종류의 qualitative와 quantitative question을 디자인 하였으며, 각각 20개의 question template과 10개의 answer template을 사용한다고 합니다. 이외에도 인간의 반올림을 모방한 human-aligned rounding mechanism등 다양한 추가 기법들을 적용하였는데, 해당 내용에 대해서는 appendix에 자세히 설명이 되어있습니다. 이를 통해 최종적으로 저자는 약 천만개의 image와 20억개의 spatial reasoning QA pair를 포함하는 거대한 데이터셋을 구축할 수 있었으며, 이를 통해 모델이 다양한 description을 학습할 수 있게 되었다고 합니다.

<br>

### 2.2. Learning Spatial Reasoning

- **Train**

&nbsp;&nbsp;&nbsp;&nbsp;이제 지금까지 생성한 데이터셋을 모델에 학습시킬 차례입니다. 학습에 사용될 아키텍쳐로는 PaLM-E를 채택하였으며, 학습 과정또한 동일하게 진행하였다고 합니다. Input으로는 image $I$와 spatial task에 대한 쿼리 $Q$를 받으며, 이에 대한 output으로는 text string 형태의 answer $A$를 출력합니다.

&nbsp;&nbsp;&nbsp;&nbsp;PaLM-E와 거의 유사한 학습 과정을 거치지만, 차이점이 몇가지 존재합니다. 우선, backbone을 기존 PaLM에서 PaLM 2-S로 수정했으며, 학습에는 PaLM-E 데이터셋과 저자들의 데이터셋을 적절히 섞어서 사용했다고 합니다. 그리고 결정적으로 SpatialVLM만의 가장 중요한 차별점은, 바로 `spatial reasoning question에 대한 답변`을 할 수 있다는 것 입니다.

<br>

- **Chain-of-Thought Spatial Reasoning**

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/54bcb149-e6d4-459c-aadc-bc5689c549ff" style="border: solid black 1px"></p>

&nbsp;&nbsp;&nbsp;&nbsp;이렇게 학습된 모델을 바탕으로 복잡한 spatial reasoning을 수행하기 위해, 독자는 `Chain-of-Thought Spatial Reasoning`이라는 새로운 메서드를 제시합니다. 해당 메서드는 real-world에서 사람이 사고하는 과정을 표방하여 만든 메서드로, 큰 문제를 작은 문제로 나누어 푸는 분할정복 알고리즘과 비슷한 동작방식을 가집니다.

&nbsp;&nbsp;&nbsp;&nbsp;조금 더 구체적으로 설명하자면, 우선 Chain-of-Thought를 위해 또 하나의 LLM(text-davinci-003)을 도입합니다. 그 이후, 위 figure에 보이는 것처럼 SpatialVLM과 계속 질문을 주고 받으며 spatial한 정보들을 얻습니다. 가령, "사진 속 음료캔들이 대충 이등변 삼각형 모양이야?"라고 질문하였을 때, LLM은 계속해서 VLM과 대화를 주고 받으며 다양한 spatial information들을 얻게되고, 최종적으로 이를 종합하여 답변하게 됩니다. 이러한 방법을 사용하게 되면, 복잡한 spatial reasoning 문제라도 쉽게 해결할 수 있다고 합니다.

<br>
<br>

## 4. Experiments

&nbsp;&nbsp;&nbsp;&nbsp;Experiments에서는 선행 연구대비 Spatial reasnoning capability 비교, 다양한 학습 전략에 따른 performance 변화, robotics등의 downtream transfer에 관한 실험들에 대해 소개하고 있습니다. 분량의 문제로 인해 해당 실험에 대해서는 따로 포스팅하지 않도록 하겠습니다. 관심 있으신 분들은 페이퍼를 참고해주세요!

<br>
<br>

## 5. Conclusion

&nbsp;&nbsp;&nbsp;&nbsp;본 페이퍼의 저자들은 VLM에 있어서 가장 challenging한 부분 중 하나였던 `spatial reasoning`과 관련된 연구를 진행하였으며, 이를 해결하기 위해 3D spatial reasoning VQA data를 자동으로 생성하는 모델을 구현하였습니다. 또한, 유한한 템플릿을 사용하여 train query들을 구축하였지만, `Chain-of-Thought Spatial Reasoning`을 통해 복잡한 문제도 해결할 수 있음을 보여주었습니다. 마지막으로, SpatialVLM은 reward annotator로서 `robotics task`에도 유용하게 활용될 수 있음을 증명하였습니다(Experiments 참고).