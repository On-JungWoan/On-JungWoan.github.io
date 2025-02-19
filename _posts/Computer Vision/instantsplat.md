---
title:  "[Arxiv] InstantSplat 리뷰"
excerpt: "InstantSplat: Sparse-view SfM-free Gaussian Splatting in Seconds (arXiv, 2024)"

categories:
  - DL_paper
tags:
  - [DL, computer_vision]

published: true

toc: true
toc_sticky: true
 
date: 2025-02-20
last_modified_at: 2025-02-20
use_math: true
---

**[Title]** InstantSplat: Sparse-view SfM-free Gaussian Splatting in Seconds

**[Keyword]** 3DGS

**[Journal]** Arxiv

**[arXiv]** <a href="https://arxiv.org/abs/2403.20309" target="blank_">https://arxiv.org/abs/2403.20309</a>

**[Summary]**

<p align="center"><video width="100%" src="https://github.com/user-attachments/assets/07b45dc6-1d90-4f14-baf2-7146d403caea" autoplay controls loop muted></video></p>

- 3DGS-based method들은 좋은 performance를 얻기 위해 필연적으로 largely overlap된 multi-view image를 필요로 함.
- but, InstantSplat은 3~12개의 매우 sparse한 view만을 사용해서 original 3DGS(300~400 view)와 비슷한 performance를 보임.
- 피팅에 걸린 시간은 1분 미만. original 3DGS가 피팅에 30분 이상, NeRF-based method들이 수 시간 걸리는 걸 생각하면 굉장히 빠른 속도임.

어떻게 이게 가능한걸까요?

<br>
<br>

## 1. Related Work

### **1-1) 3D Gaussian Splatting for Real-Time Radiance Field Rendering (SIGGRAPH, 2023)**

![Image](https://github.com/user-attachments/assets/6de471d3-9dda-4e62-81c1-44aa27c5912e)

&nbsp;&nbsp;&nbsp;&nbsp;먼저, InstantSplat이 original 3DGS와 어떤 점이 다른지 알아보도록 하겠습니다. 위 figure는 3DGS의 전체적인 flow를 나타내고 있습니다. 3DGS에서는 gaussian과 camera를 initialize하기 위해 SfM 계열의 COLMAP을 사용하고 있으며, COLMAP으로부터 얻은 sparse point cloud로 가우시안의 mean값을 초기화합니다. 이후, init GS와 camera pose를 rasterization func.에 넣어서 2D image로 rendering 시켜주며 GT image와 photometric loss를 걸어주어 가우시안의 파라미터들을 optimize합니다.

<br>

![Image](https://github.com/user-attachments/assets/cc3be29c-43cc-4f0a-9fcc-5ae72fde8d98)

&nbsp;&nbsp;&nbsp;&nbsp;이 때, gaussian이 local minima에 빠지지 않도록 3DGS만의 특별한 trick인 ADC(Adaptive Density Control)를 적용합니다. target scene을 제대로 표현하지 못하는 small gaussian에 대해서는 해당 gaussian의 gradient방향으로 또다른 gaussian을 clone해주며, 특정 영역을 과도하게 가리는 large gaussian의 경우 1.6배 더 작은 2개의 gaussian으로 split해줍니다. 추가적으로, gaussian의 gradient가 특정 thold 미만인 saturated gaussian에 대해서는 opacity를 0으로 reset해주게 되며, opacity가 0인 gaussian에 대해서는 pruning을 진행해줍니다. 이러한 일련의 과정을 특정 iteration마다 반복해서 진행해줌으로써, gaussian들이 local minima에 빠지지 않도록 합니다.

<br>

![Image](https://github.com/user-attachments/assets/02ea7a44-2786-4f69-bcdb-9abc4080b07f)

&nbsp;&nbsp;&nbsp;&nbsp;하지만, ADC는 hyper parameter(num step, size thold 등등)를 어떻게 튜닝하느냐에 따라 최종 output performance가 크게 변하는 매우 hueristic한 process입니다(위 테이블 1번 row). 또한, 3DGS의 optimization process는 SfM의 init 결과에 매우 sensitive합니다(위 테이블 2,3번 row). 따라서 좋은 reconstruction quality를 위해서는 init 결과를 잘 뽑는게 중요한데, 이를 위해 필연적으로 largely overlaping된 multi-view seq가 필요합니다. 또한, SfM으로부터 pcd/camera를 얻는 과정은 이미지 개수/resolution 등에 따라 수 시간 ~ 많게는 일주일 이상 소요되는 매우 time-consuming한 process입니다.

<br>

![Image](https://github.com/user-attachments/assets/1f96ef6b-1f36-4685-9413-7aeb81d38561)

&nbsp;&nbsp;&nbsp;&nbsp;InstantSplat의 저자는 이러한 비효율적인 SfM 대신, DUSt3R-based의 off-the-shelf model을 사용하여 매우 적은 수의 viewpoint만으로도 가우시안을 피팅시킬 수 있었다고 합니다. 또한, hueristic한 ADC 과정은 아예 생략해버렸고, 결과적으로 200~1000 step만으로도 가우시안을 피팅시킬 수 있었다고 합니다.

> 이게 가능한 이유가 off-the-shelf model로부터 뽑은 point cloud가 정확하다는 가정이 깔려있기 때문인 것 같은데, DUSt3R가 fail한 경우에는 최종 performance에 얼마나 영향이 있을지 궁금하네요. densification을 하지 않으면 오히려 init값에 더 sensitive할 것 같은데, off-the-shelf에 너무 의존적인 구조인건 아닐까 싶습니다.

<br>

### **1-2) DUSt3R: Geometric 3D Vision Made Easy (CVPR, 2024)**

그렇다면 앞서 언급된 DUSt3R는 무엇을 하는 모델일까요?

![Image](https://github.com/user-attachments/assets/d5ff7784-dcf4-4cef-aace-9b80be4eb103)

&nbsp;&nbsp;&nbsp;&nbsp;DUSt3R(CVPR, 2024)는 SfM과 비슷하게 input image pair에 대해 point cloud와 camera pose를 찾아주는 method입니다. SfM과는 달리 image pair간에 overlap된 부분이 많이 없어도 어느정도 동작하며, 2장의 image pair를 처리하는 데 특화되어있는 모델입니다.

<br>

![Image](https://github.com/user-attachments/assets/e4e81f97-7523-4ced-ab1c-72db8c4630b7)

&nbsp;&nbsp;&nbsp;&nbsp;DUSt3R는 크게 (1)basic model 파트와, (1)에서 얻은 output을 사용한 (2)downstream aplication 파트로 나누어져 있습니다.

<br>

![Image](https://github.com/user-attachments/assets/17a03375-bf5c-4401-9838-0da3c850ff56)

&nbsp;&nbsp;&nbsp;&nbsp;(1)basic model은 2개의 ViT encoder <-> Transformer Decoder가 병렬로 이루어져 있고, 각 head는 3D point cloud와 이에 대한 confidence score를 예측하도록 학습되어 있습니다(자세한 학습 과정에 대해서는 본 포스팅에서는 다루지 않겠습니다). 이 떄 transformer decoder는, 자기 자신의 이미지로부터 얻은 feature에 대해서는 self attention을, 참조 이미지로부터 얻은 feature와는 cross attention을 진행해줌으로써 두 이미지 pair사이의 관계성을 학습할 수 있었다고 합니다. 하지만, basic model이 뽑은 point들은 global한 좌표계에 align되어 있지 않기 때문에 이를 align해주는 추가적인 과정이 필요합니다. 또한, 이 상태에서는 camera pose또한 알 수 없기 때문에 point cloud로부터 camera pose를 찾는 과정도 필요합니다. 이러한 과정이 (2)downstream aplication에서 진행됩니다. 이외에도 더 많은 process들이 있지만 본 포스팅에서는 간단히 필요한 부분들만 다루도록 하겠습니다.

<br>

- Downstream Applications: Relative camera pose estimation

![Image](https://github.com/user-attachments/assets/11711776-f577-4966-9a26-c171a9e382ff)

![Image](https://github.com/user-attachments/assets/6b08cdd3-1d5b-4959-bffd-3c3d133c6348)

&nbsp;&nbsp;&nbsp;&nbsp;먼저, 카메라 pose(extrinsic)를 찾는 과정입니다. 해당 step의 목표는 빨간색 카메라를 원점에 두고, 해당 카메라에 대한 파란색 카메라의 상대적인 pose를 구하는 것 입니다. 이를 위해, Image 1으로부터 얻은 point cloud($X^{1,1}$)의 origin에 Camera 1을 initialize합니다. 이후, point cloud 1(빨간색)과 point cloud 2(파란색) 모두를 Camera 2에 대한 좌표계로 표현합니다($X^{1,2}$, $X^{2,2}$ -> 두 포인트의 원점이 camera2가 되는 것임). 마지막으로, $X^{2,2}$와 $X^{1,2}$가 최대한 가까워지도록 하는 Rotaion $R$과 translation $t$, scale $\sigma$를 구해줌으로써, camera 2의 relative pose를 구할 수 있게 됩니다.

<br>

- Downstream Applications: Recovering intrinsics

![Image](https://github.com/user-attachments/assets/a88e2981-4e87-4c89-b44a-4a2e13b192e2)

&nbsp;&nbsp;&nbsp;&nbsp;다음으로, camera intrinsic parameter를 찾는 과정입니다. intrinsic parameter는 크게 principal point (x,y)와 focal length (f)로 이루어져 있습니다(보통 2D shear는 고려 x). principal point는 pinhole camera에서 optical center에 대한 2D 좌표값이며, focal length는 optical center에서 image plane까지의 거리를 나타냅니다.

<br>
<br>
<br>

![Image](https://github.com/user-attachments/assets/92fac98c-ae1f-4991-8c11-a87b80d2c197)