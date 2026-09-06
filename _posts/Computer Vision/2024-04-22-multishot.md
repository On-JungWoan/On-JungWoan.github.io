---
title:  "[CVPR 2022] Human Mesh Recovery from Multiple Shots 논문 리뷰"
excerpt: "multishot: Human Mesh Recovery from Multiple Shots"

categories:
  - papers
tags:
  - "DL"
  - "computer_vision"
  - "human-motion"
  - "3d-reconstruction"

published: true

toc: true
toc_sticky: true
 
date: 2024-04-22
last_modified_at: 2024-04-22
use_math: true
permalink: "/dl_paper/multishot/"
legacy_category: DL_paper
---

> 논문링크 : <a href="https://arxiv.org/pdf/2012.09843.pdf" target="blank_">Human Mesh Recovery from Multiple Shots</a>

> Implementation : <a href="https://github.com/geopavlakos/multishot" target="blank_">https://github.com/geopavlakos/multishot</a>

오늘 소개드릴 페이퍼는 CVPR 2022에 accept된 `Human Mesh Recovery from Multiple Shots`입니다.

<br>
<br>

## 1. Abstract & Introduction

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/a6b6f86b-76d4-4e04-ba9f-4cfe4ada20cf" style="border: solid black 1px"></p>

&nbsp;&nbsp;&nbsp;&nbsp;영화와 같은 edited dataset은 매우 풍부한 human pose 및 interaction을 담고 있지만, 학습 데이터로는 아직 활발히 사용되지 않고 있습니다. edited dataset을 학습 데이터 셋으로 활용하지 못하는 가장 큰 이유로는 `shot change`와 `close up`이 있습니다. 영화에서 shot은 끊임없이 바뀌며, 때로는 인물을 과하게 close up하는 장면도 등장합니다. 기존 모델들은 이러한 edited dataset의 한계점을 극복하지 못하였기 떄문에, 해당 데이터셋의 이점을 제대로 활용하지 못하였습니다.

&nbsp;&nbsp;&nbsp;&nbsp;하지만, shot이 수시로 바뀌고 인물이 클로즈업 되더라도 world coordinate 위에서 해당 `인물의 global한 location은 변함이 없습니다`(global coordinates상에서 human은 가만히 있고 카메라만 움직이기 때문). 본 paper의 저자들은 이러한 인사이트를 활용하여 edited dataset의 한계점을 극복하였습니다. 따라서 최종적으로 AVA라는 movie dataset으로부터 human mesh에 대한 pseudo GT를 만들어낼 수 있었으며, 이를 `MS-AVA`라고 명명하였습니다. 이후 MS-AVA를 사용하여 HMR과 transformer를 학습시킴으로써, movie로부터 direct하게 human mesh를 뽑아낼 수 있었으며, `target person에만 attention` 할 수 있었다고 합니다.

<br>
<br>

## 2. Multi-shot optimization

&nbsp;&nbsp;&nbsp;&nbsp;먼저, AVA dataset으로부터 pseudo GT를 추출하는 과정에 대해 설명드리도록 하겠습니다.

### 2.1. Preprocessing steps

&nbsp;&nbsp;&nbsp;&nbsp;우선, 저자들은 `openpose`를 사용하여 input video로부터 2D keypoint에 대한 tracklet을 얻습니다. openpose로부터 얻은 output이 꽤 준수한 편이긴 하지만, shot이 바뀌는 케이스에서는 tracking이 끊기게 되는 문제가 있었다고 합니다. 따라서 좀 더 long distance의 tracklet을 얻기 위해, 우선 `shot detection 알고리즘`으로 shot change를 detection 해줍니다. 이후, `human re-identification network`를 사용하여 shot 이후 잘못 matching된 tracklet을 연결시켜줍니다. 이로서, 저자들은 shot boundary를 넘어서는 `long distance의 tracklet`을 얻을 수 있었다고 합니다.

<br>

### 2.2. Multi-shot optimization

&nbsp;&nbsp;&nbsp;&nbsp;이후에는 위에서 얻은 tracklet을 사용하여 SMPL parameter를 optimize 해줍니다. Loss는 다음과 같습니다.

<p align="center"><img width="60%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/9863e555-7927-44fb-9b74-10c3fb1d50d8"></p>

&nbsp;&nbsp;&nbsp;&nbsp;먼저, 앞의 projection term과 prior term은 SMPL-X의 objective와 동일합니다. projection term은, SMPL로부터 얻은 mesh를 2D joint로 projection 시킨 결과와 openpose로 얻은 gt keypoint간의 l2 loss를 취해줌으로서 얻어지는 value입니다. prior term은 body pose와 motion에 대한 prior를 담고 있습니다.

&nbsp;&nbsp;&nbsp;&nbsp;그 다음으로 등장하는 $E_{sm}$가 해당 페이퍼의 key contribution이라고 할 수 있습니다. 해당 term은, shot이 바뀔 때 앞 뒤 frame간 joint(혹은 SMPL param)에 smoothing을 해주는 loss인데, 한 가지 trick이 존재합니다. 해당 term을 camera frame에서 바로 사용할 경우, 갑작스러운 shot change로 인해 매우 큰 loss를 갖게 될 것이며, 제대로 smoothing이 되지 않을 것입니다. 따라서 저자는 이를 해결하기 위해, 먼저 PHALP를 사용하여 해당 frame으로부터 SMPL parameter를 얻으며, pose parameter로부터 `global orientation`(0번 joint에 대한 pose param.) matrix를 추출합니다. global orientation은 root joint에 대한 ratation matrix이기 떄문에, global orientation의 inverse matrix를 body joint에 곱해줌으로써 `keypoint를 한 방향으로 align`할 수 있습니다.


<p align="center"><img width="80%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/780a1afe-e66c-42a1-8b36-c773ed342b58" style="border: solid black 1px"></p>

<br>

&nbsp;&nbsp;&nbsp;&nbsp;이렇게 얻어진 keypoint($X_{can}$)와 pose parameter의 앞 뒤 frame에 L2 loss를 걸어줌으로써, shot boundary의 keypoint에 대해 smoothing을 진행합니다.

<p align="center"><img width="40%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/cc9616e4-3514-4687-9d40-a9fbef85e7a7"></p>

<br>

### 2.3. Multi-Shot AVA dataset

&nbsp;&nbsp;&nbsp;&nbsp;위에서 설명한 방법을 AVA movie dataset에 적용하여 human mesh에 대한 pseudo GT를 생성합니다. 이로써, 방대한 양의 영화 데이터셋을 학습 데이터셋으로 사용할 수 있게 되었으며, 저자들은 이 새로운 데이터 셋을 `Multi Shot AVA(MS-AVA)`라고 소개하였습니다.


<br>
<br>


## 3. Human Mesh Recovery

&nbsp;&nbsp;&nbsp;&nbsp;이후, 저자는 해당 데이터셋 활용해 총 2가지의 input case에 대한 모델을 학습시켰습니다. 각각에 대해서는 다음 섹션에서 설명드리도록 하겠습니다.

<p align="center"><img width="80%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/58e3c904-1156-4c89-a42f-8217234cdb57" style="border: solid black 1px"></p>

<br>

### 3-1. Single-frame Model

&nbsp;&nbsp;&nbsp;&nbsp;먼저, single-frame model입니다. input으로 single image를 받는 model이며, `HMR`을 사용하여 학습하였습니다. 크게 특별한 점은 없고, 2D joint와 SMPL Parameter에 대한 loss를 걸어주었습니다.

<p align="center"><img width="40%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/2a69b2a3-fd7a-49fc-878b-ba1b928716b9"></p>


<br>

### 3-2. Temporal Model

&nbsp;&nbsp;&nbsp;&nbsp;다음으로는 temporal model입니다. video input을 받기 위해 `pure한 transformer`를 사용하였습니다. movie dataset의 경우, shot이 변함에 따라 tracking하지 않는 인물이 갑작스럽게 등장할 수 있는데, transformer의 attention 매커니즘을 사용하여 `tracking하고자 하는 인물에만 attention`할 수 있습니다.

<p align="center"><img width="80%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/cebfb7ab-07d9-4906-9786-e8fac14b0292" style="border: solid black 1px"></p>


<br>
<br>

## 4. Experiment

### 4-1. Multi-shot optimization

&nbsp;&nbsp;&nbsp;&nbsp;`Multi-shot optimization`이 single shot을 사용하는 다른 optimization 방법들보다 `더 좋은 performance`를 보여주었다고 합니다.

<p align="center"><img width="50%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/6ba4b95c-5473-4582-a358-1bb0c1205eba" style="border: solid black 1px"></p>

<p align="center"><img width="50%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/b7443461-4434-48a2-abcf-e284da3e6cc3" style="border: solid black 1px"></p>

<br>

### 4-2. Single-frame model

&nbsp;&nbsp;&nbsp;&nbsp;MS-AVA dataset을 사용하여 학습한 HMR이 original HMR대비 훨씬 더 좋은 퍼포먼스를 보입니다. `PA-MPJPE` 기준, 에러가 `20mm이상 감소`한 것을 확인할 수 있습니다.

<p align="center"><img width="50%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/baffacd6-6f5a-4877-a92b-0150b5150677" style="border: solid black 1px"></p>

<p align="center"><img width="50%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/13b2e6c2-d253-40ee-bf10-89068b319b01" style="border: solid black 1px"></p>

<br>

### 4-2. Temporal model

&nbsp;&nbsp;&nbsp;&nbsp;각기 다른 encoder를 사용하여 모델을 train하였을 떄의 결과입니다. 3가지 encoder 모두 결국에는 좋은 performance를 보였지만, transformer의 수렴 속도가 가장 빠르고 stable하다고 합니다.

<p align="center"><img width="50%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/3ef6653f-8bb4-4326-9b4d-2ec2729adb8a" style="border: solid black 1px"></p>

<p align="center"><img width="50%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/87cbdc38-139a-4217-b535-14b475809108" style="border: solid black 1px"></p>