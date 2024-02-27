---
title:  "[private] SpatialVLM(2024) 논문 리뷰"
excerpt: "Spatial VLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities"

categories:
  - DL_paper
tags:
  - [DL, computer_vision]

published: true

toc: true
toc_sticky: true
 
date: 2024-02-21
last_modified_at: 2024-02-21
use_math: true
---

> 논문링크 : <a href="https://arxiv.org/pdf/2401.12168.pdf" target="blank_">Spatial VLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities</a>

> project page : <a href="https://spatial-vlm.github.io/" target="blank_">https://spatial-vlm.github.io/</a>

오늘 소개드릴 페이퍼는 `Spatial VLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities`입니다. 구글 딥마인드에서 2024년 1월에 publish하였으며, CVPR 2024에 accept 되었습니다.

<br>
<br>

## 1. Abstract

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/717fffd8-0c13-4b1f-b09b-40299272112a" style="border: solid black 1px"></p>

&nbsp;&nbsp;&nbsp;&nbsp;VQA와 robotics 분야에 있어서 spatial relationship에 대해 이해하는 것은 상당히 중요합니다. CLIP등으로 대표되는 VLM이 최근 VQA 벤치마크에서 주목할만한 성능을 보였으나, 여전히 3D space에 대한 이해능력은 많이 뒤떨어지는 실정입니다. 위 figure에서 언급된 바와 같이, 모델은 input image에서 baseball player와 black man 사이의 거리를 인지하지 못합니다. 저자는 이러한 한계의 원인이 3D spatial knowledge의 부족에 있다고 하였으며, 이러한 문제를 인터넷에서 수집된 데이터만을 사용하여 해결하려는 것이 문제라고 밝히고 있습니다.

&nbsp;&nbsp;&nbsp;&nbsp;해당 연구에서는, 이러한 문제를 해결하기 위해 3D spatial VQA 데이터를 automatic하게 generation하는 framework를 개발하였으며, 그 결과 20억개의 VQA example을 1000만개의 real-world image로 scale up하는 데 성공했다고 합니다. 이러한 데이터셋을 사용하여 VLM을 훈련함으로써, spatial VQA에 대한 질적/양적인 성능을 상당히 끌어올릴 수 있었으며, spatial reasoning과 robotics 분야로의 새로운 downstream applications을 가능케하였다고 합니다.

<br>
<br>

## 2. Introduction

&nbsp;&nbsp;&nbsp;&nbsp;최근 image captioning, VQA, action recognition 등의 다양한 분야에서, VLM이 상당한 두각을 나타내고 있습니다. 하지만 VLM의 눈부신 성장과는 대조적으로, VLM이 고질적으로 겪고있는 문제가 있습니다. 그것은 바로 SOTA VLM들의 spatial 정보에 대한 이해 부족입니다. 가령, 3D space상의 다양한 object들의 spatial한 관계라던지, 해당 object의 position에 대한 이해를 필요로 하는 task에서는 VLM이 상당히 약한 모습을 보이고 있습니다. 이러한 spatial information을 이해하는 능력은 그 자체로도 매우 쓸모있을 뿐만 아니라, robotics나 AR등의 downstream task에서도 유용하게 사용될 수 있습니다. 따라서 이러한 limitation을 해결하는 것이 VLM에 있어서 가장 중요한 도전과제라고 할 수 있습니다.

<br>
<br>

## 3. Method

### 3.1. Spatial Grounding from 2D Images

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/5b9c1b36-d838-45ae-9c5f-a5ec4b6d6e88" style="border: solid black 1px"></p>

&nbsp;&nbsp;&nbsp;&nbsp;우선, 저자들은 최근 VLM의 공간 추론능력에 대한 한계점이 모델의 구조 때문이 아니라, spatial reasoning에 대한 학습 데이터가 부족했기 때문일 거라고 추측하고 있습니다. 따라서, 이러한 문제점을 해결하기 위해 spatial reasoning에 대한 질의를 포함하고 있는 VQA data를 생성하는 것부터가 해당 연구의 첫 걸음이라고 할 수 있습니다. 해당 연구에서는 이를 위해 VQA 데이터를 automatic하게 generate하는 pipeline을 구현했다고 소개하고 있습니다. 이에 대한 내용이 위 figure에 간략히 표현되어 있습니다. 이제부터는 figure의 각 단계에 대해 조금 더 자세히 설명드려보도록 하겠습니다.

<br>

- **(a) Semantic Filtering**

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/d09651cb-1649-43df-8f05-b02b3252a26f" style="border: solid black 1px"></p>

&nbsp;&nbsp;&nbsp;&nbsp;generation의 첫 단계는, 필요 없는 이미지를 필터링하는 과정으로부터 시작됩니다. 인터넷에서 수집된 데이터셋은 VLM의 학습에 광범위하게 사용되지만, spatial한 정보를 학습하기 위해서는 이러한 데이터셋을 바로 학습할 수 없습니다. 인터넷에 돌아다니는 이미지의 대부분은 배경이 없거나, 딱 하나의 object만을 포함하고 있는 경우가 많습니다. 실제로 구글에 ketchup에 대한 이미지를 검색했을 때, 위와 같이 background가 없거나 single object에 대한 이미지인 경우가 대다수입니다.

&nbsp;&nbsp;&nbsp;&nbsp;이러한 이미지에서는 어떠한 spatial한 단서를 얻을 수 없기 떄문에(인간조차도 이러한 이미지에 대해서는 어떠한 단서도 얻을 수 없음), 우선적으로 해당 이미지는 학습 데이터셋에서 제외하는 과정을 거칩니다. 이를 위해 CLIP base의 open-vocabulary classification model을 사용함으로써 모든 이미지를 분류하고, 필요없는 이미지는 필터링합니다.

<br>

- **(b) Object-centric Contexts Extraction from 2D Images**

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/1a2e07ba-9a73-4154-b050-86f4992cbb0c" style="border: solid black 1px"></p>

&nbsp;&nbsp;&nbsp;&nbsp;이후에는 오브젝트와 관련된 정보를 얻기위해 다양한 작업을 수행합니다. 해당 과정에는 Region Captioning, Depth Estimation, Segmentation 등의 작업이 포함되며, 결과적으로 2D image로부터 object-centric의 context를 얻게됩니다.

&nbsp;&nbsp;&nbsp;&nbsp;그리고 camera calibration에 관한 내용이 Appendix에 언급되어 있습니다. 여기서 특이한 점은, 카메라 파라미터를 사용하지 않고 projection을 진행한다는 점입니다. 이를 위해, 작은 segmentation model을 학습시켜서 floor나 table top등을 찾아내고, 찾아낸 평면에 대해 camera origin을 projection함으로써 point cloud들을 world coordinate에 projection시킨다고 합니다. 이에 대한 내용이 위 pseudo-code에 자세히 나와 있습니다.


<br>

- **(c) Lifting 2D Contexts to 3D Contexts**

&nbsp;&nbsp;&nbsp;&nbsp;기존 object detection이나 bounding box positioning을 사용하여 얻어진 VQA dataset은 2D(and pixel-level)의 reasoning 수준에 머물러 있습니다. 그 이유는 depth, spatial information이 포함된 context가 부족하기 떄문입니다. 따라서 저자들은 이러한 문제점을 해결하기 위해, 우선 2D pixel을 metric-scale의 3D point cloud로 lift 해주었습니다(by depth estimation). 그 이후에는, 위에서 잠깐 언급한 바와 같이, segmentation model을 사용하여 point cloud들을 camera coordinates에서 world coordinate로 projection해주었다고 합니다.

<br>

- **(d) Ambiguity Resolution**

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/02aff9a2-f8a3-4a5c-afcf-4bee64cfb420" style="border: solid black 1px"></p>


&nbsp;&nbsp;&nbsp;&nbsp;종종 하나의 이미지 안에, 비슷한 카테고리를 가지는 object가 여러개 있을 수 있는데, 이 경우 모델이 하나의 caption에 대해 여러개의 object를 참조하게 되는 ambiguity가 발생할 수 있다고 합니다. 이를 해결하기 위해, SpatialVLM의 저자들은 다음과 같이 2개의 implementation을 고려하였습니다.

  1. 일반적인 object detector의 사용 지양
     - 이러한 detector를 사용할 경우, 단순한 카테고리(ex. cake)만을 produce하는 경향이 있다고 합니다.
     - 따라서 저자들은 FlexCap이라는 object-centric captioning approach를 detector로 채택하였습니다.
  2. 추가적인 post-processing algorithm 구현
     - 이외에도 Ambiguity를 해결하기 위해 추가적인 augment, remove과정을 거칩니다.
     - 이를 위해, CLIP을 사용하여 similarity를 계산하고, 이 similarity가 특정 threshold 이상이면 유사한 caption으로 판단합니다.
     - 만약 유사한 caption이,
       - `정확히 2개 존재하는 그룹`이라면, 'to the left'와 같은 spatial attribute를 caption에 추가해주는 augment 과정을 거칩니다.
       - `2개보다 많이 존재하는 그룹`이라면, ambiguity를 제거하기 위해 해당 caption은 delete합니다.

<br>

### 3.2. Large-Scale Spatial Reasoning VQA Dataset

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/77d189af-adcb-4420-b303-e2e93829a7e7" style="border: solid black 1px"></p>

&nbsp;&nbsp;&nbsp;&nbsp;지금까지 일련의 과정을 모두 끝마친 뒤, 본격적으로 VLM에 간단한 공간 추론 능력을 학습시키기 위해 데이터셋을 구축해야 합니다. 이를 위해 저자는 spatial-reasoning QA pair를 갖는 데이터셋을 생성하였으며, 각 pair는 2개 이하의 object에 대한 정보를 포함합니다. 이에 대한 전체적인 개요를 확인하고 싶으신 분은, [#3.1.](#31-spatial-grounding-from-2d-images)의 figure에서 (e) 부분을 참고하시면 됩니다.

&nbsp;&nbsp;&nbsp;&nbsp;Answer의 경우는 해당 페이퍼의 저자가 개발한 적절한 function을 통해 생성되며, input으로는 3D bounding box와 point cloud를 받습니다. Question의 경우는 다음과 같이 2개의 카테고리로 나누어집니다.

- **Qualitative questions**
  - 아래 예시와 같이, spatial relation에 대한 판단을 요구하는 질의입니다.
  - "의자가 오븐 앞에 위치해 있어?", "접시가 냅킨 오른쪽에 있어 왼쪽에 있어?"

- **Quantitative questions**
  - 아래 예시와 같이, 구체적인 수치를 요구하는 질의입니다.
  - "cake모양의 집과 보라색 옷을 입은 여자 사이의 거리를 측정해 줘"

&nbsp;&nbsp;&nbsp;&nbsp;최종적으로 저자는 약 천만개의 image와 20억개의 spatial reasoning QA pair를 포함하는 거대한 데이터셋을 구축할 수 있었으며, 이를 통해 모델이 다양한 description을 학습할 수 있게 되었다고 합니다.

<br>

### 3.3. Learning Spatial Reasoning

- **Train**

&nbsp;&nbsp;&nbsp;&nbsp;이제 지금까지 생성한 데이터셋을 모델에 학습시킬 차례입니다. 학습에 사용될 아키텍쳐로는 PaLM-E를 채택하였으며, 학습 과정또한 동일하게 진행하였다고 합니다. Input으로는 image $I$와 spatial task에 대한 쿼리 $Q$를 받으며, 이에 대한 output으로는 text string 형태의 answer $A$를 출력합니다.

&nbsp;&nbsp;&nbsp;&nbsp;PaLM-E와 거의 유사한 학습 과정을 거치지만, 차이점이 몇가지 존재합니다. 우선, backbone을 기존 PaLM에서 PaLM 2-S로 수정했으며, 학습에는 PaLM-E 데이터셋과 저자들의 데이터셋을 적절히 섞어서 사용했다고 합니다. 그리고 결정적으로 SpatialVLM만의 가장 중요한 차별점은, 바로 spatial reasoning question에 대한 답변을 할 수 있다는 것 입니다.

<br>

- **Chain-of-Thought Spatial Reasoning**

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/54bcb149-e6d4-459c-aadc-bc5689c549ff" style="border: solid black 1px"></p>

&nbsp;&nbsp;&nbsp;&nbsp;이렇게 학습된 모델을 바탕으로 복잡한 spatial reasoning을 수행하기 위해, 독자는 `Chain-of-Thought Spatial Reasoning`이라는 새로운 메서드를 제시합니다. 해당 메서드는 real-world에서 사람이 사고하는 과정을 표방하여 만든 메서드로, 큰 문제를 작은 문제로 나누어 푸는 분할정복 알고리즘과 비슷한 동작방식을 가집니다.

&nbsp;&nbsp;&nbsp;&nbsp;조금 더 구체적으로 설명하자면, 우선 Chain-of-Thought를 위해 또 하나의 LLM(text-davinci-003)을 도입합니다. 그 이후, 위 figure에 보이는 것처럼 SpatialVLM과 계속 질문을 주고 받으며 spatial한 정보들을 얻습니다. 가령, "사진 속 음료캔들이 대충 이등변 삼각형 모양이야?"라고 질문하였을 때, LLM은 계속해서 VLM과 대화를 주고 받으며 다양한 spatial information들을 얻게되고, 최종적으로 이를 종합하여 답변하게 됩니다. 이러한 방법을 사용하게 되면, 복잡한 spatial reasoning 문제라도 쉽게 해결할 수 있다고 합니다.

<br>
<br>

## 4. Experiments

<img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/908093f2-ad08-4162-866a-2df074b03090">

<img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/461109f2-859e-4b9a-ac4e-009ad19059b4">

기존 선행연구들 대비 SpatialVLM이 높은 마진으로 더 좋은 spatial reasoning performance를 보이는 것을 확인할 수 있습니다.