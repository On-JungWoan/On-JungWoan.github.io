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

구글 딥마인드에서 2024년 1월에 publish한 논문입니다.

<br>
<br>

# 1. Abstract

<img width="840" alt="image" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/717fffd8-0c13-4b1f-b09b-40299272112a">

&nbsp;&nbsp;&nbsp;&nbsp;VQA와 robotics 분야에 있어서 spatial relationship에 대해 이해하는 것은 상당히 중요합니다. CLIP등으로 대표되는 VLM이 최근 VQA 벤치마크에서 주목할만한 성능을 보였으나, 여전히 3D space에 대한 이해능력은 많이 뒤떨어지는 실정입니다. 위 figure에서 언급된 바와 같이, 모델은 input image에서 baseball player와 black man 사이의 거리를 인지하지 못합니다. 저자는 이러한 한계의 원인이 3D spatial knowledge의 부족에 있다고 하였으며, 이러한 문제를 인터넷에서 수집된 데이터만을 사용하여 해결하려는 것이 문제라고 밝히고 있습니다.

&nbsp;&nbsp;&nbsp;&nbsp;해당 연구에서는, 이러한 문제를 해결하기 위해 3D spatial VQA 데이터를 automatic하게 generation하는 framework를 개발하였으며, 그 결과 20억개의 VQA example을 1000만개의 real-world image로 scale up하는 데 성공했다고 합니다. 이러한 데이터셋을 사용하여 VLM을 훈련함으로써, spatial VQA에 대한 질적/양적인 성능을 상당히 끌어올릴 수 있었으며, spatial reasoning과 robotics 분야로의 새로운 downstream applications을 가능케하였다고 합니다.

<br>
<br>

# 2. Introduction

&nbsp;&nbsp;&nbsp;&nbsp;최근 image captioning, VQA, action recognition 등의 다양한 분야에서, VLM이 상당한 두각을 나타내고 있습니다. 하지만 VLM의 눈부신 성장과는 대조적으로, VLM이 고질적으로 겪고있는 문제가 있습니다. 그것은 바로 SOTA VLM들의 spatial 정보에 대한 이해 부족입니다. 가령, 3D space상의 다양한 object들의 spatial한 관계라던지, 해당 object의 position에 대한 이해를 필요로 하는 task에서는 VLM이 상당히 약한 모습을 보이고 있습니다. 이러한 spatial information을 이해하는 능력은 그 자체로도 매우 쓸모있을 뿐만 아니라, robotics나 AR등의 downstream task에서도 유용하게 사용될 수 있습니다. 따라서 이러한 limitation을 해결하는 것이 VLM에 있어서 가장 중요한 도전과제라고 할 수 있습니다.

<br>
<br>

# 3. Method

## 3.1. Spatial Grounding from 2D Images

<img width="935" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/5b9c1b36-d838-45ae-9c5f-a5ec4b6d6e88">

&nbsp;&nbsp;&nbsp;&nbsp;우선, 저자들은 최근 VLM의 공간 추론능력에 대한 한계점이 모델의 구조 때문이 아니라, spatial reasoning에 대한 학습 데이터가 부족했기 때문일 거라고 추측하고 있습니다. 따라서, 이러한 문제점을 해결하기 위해 spatial reasoning에 대한 질의를 포함하고 있는 VQA data를 생성하는 것부터가 해당 연구의 첫 걸음이라고 할 수 있습니다. 해당 연구에서는 이를 위해 VQA 데이터를 automatic하게 generate하는 pipeline을 구현했다고 소개하고 있습니다. 이에 대한 내용이 위 figure에 간략히 표현되어 있습니다. 이제부터는 figure의 각 단계에 대해 조금 더 자세히 설명드려보도록 하겠습니다.

<br>

- **(a) Semantic Filtering**

<img width="1059" alt="image" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/d09651cb-1649-43df-8f05-b02b3252a26f">

&nbsp;&nbsp;&nbsp;&nbsp;generation의 첫 단계는, 필요 없는 이미지를 필터링하는 과정으로부터 시작됩니다. 인터넷에서 수집된 데이터셋은 VLM의 학습에 광범위하게 사용되지만, spatial한 정보를 학습하기 위해서는 이러한 데이터셋을 바로 학습할 수 없습니다. 인터넷에 돌아다니는 이미지의 대부분은 배경이 없거나, 딱 하나의 object만을 포함하고 있는 경우가 많습니다. 실제로 구글에 ketchup에 대한 이미지를 검색했을 때, 위와 같이 background가 없거나 single object에 대한 이미지인 경우가 대다수입니다.

&nbsp;&nbsp;&nbsp;&nbsp;이러한 이미지에서는 어떠한 spatial한 단서를 얻을 수 없기 떄문에(인간조차도 이러한 이미지에 대해서는 어떠한 단서도 얻을 수 없음), 우선적으로 해당 이미지는 학습 데이터셋에서 제외하는 과정을 거칩니다. 이를 위해 CLIP base의 open-vocabulary classification model을 사용함으로써 모든 이미지를 분류하고, 필요없는 이미지는 필터링합니다.

<br>

- **(b) Object-centric Contexts Extraction from 2D Images**

&nbsp;&nbsp;&nbsp;&nbsp;이후에는 오브젝트와 관련된 정보를 얻기위해 다양한 작업을 수행합니다. 해당 과정에는 Region Captioning, Depth Estimation, Segmentation 등의 작업이 포함되며, 결과적으로 2D image로부터 object-centric의 context를 얻게됩니다.

&nbsp;&nbsp;&nbsp;&nbsp;그리고 camera calibration에 관한 내용이 Appendix에 언급되어 있습니다. 여기서 특이한 점은, 카메라 파라미터를 사용하지 않고 projection을 진행한다는 점입니다. 이를 위해, 작은 segmentation model을 학습시켜서 floor나 table top등을 찾아내고, 찾아낸 평면에 대해 camera origin을 projection함으로써 point cloud들을 world coordinate에 projection시킨다고 합니다. 이에 대한 내용이 아래 pseudo-code에 자세히 나와 있습니다.

<details>
<summary><b>pseudo-code</b></summary>
<div class="colorscripter-code" style="color:#f0f0f0;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important; position:relative !important;overflow:auto"><table class="colorscripter-code-table" style="margin:0;padding:0;border:none;background-color:#272727;border-radius:4px;" cellspacing="0" cellpadding="0"><tr><td style="padding:6px;border-right:2px solid #4f4f4f"><div style="margin:0;padding:0;word-break:normal;text-align:right;color:#aaa;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;line-height:130%"><div style="line-height:130%">1</div><div style="line-height:130%">2</div><div style="line-height:130%">3</div><div style="line-height:130%">4</div><div style="line-height:130%">5</div><div style="line-height:130%">6</div><div style="line-height:130%">7</div><div style="line-height:130%">8</div><div style="line-height:130%">9</div><div style="line-height:130%">10</div><div style="line-height:130%">11</div><div style="line-height:130%">12</div><div style="line-height:130%">13</div><div style="line-height:130%">14</div><div style="line-height:130%">15</div><div style="line-height:130%">16</div><div style="line-height:130%">17</div><div style="line-height:130%">18</div><div style="line-height:130%">19</div><div style="line-height:130%">20</div><div style="line-height:130%">21</div><div style="line-height:130%">22</div><div style="line-height:130%">23</div><div style="line-height:130%">24</div><div style="line-height:130%">25</div><div style="line-height:130%">26</div><div style="line-height:130%">27</div><div style="line-height:130%">28</div></div></td><td style="padding:6px 0;text-align:left"><div style="margin:0;padding:0;color:#f0f0f0;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;line-height:130%"><div style="padding:0 6px; white-space:pre; line-height:130%">Input:</div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%">depth:&nbsp;predicted&nbsp;depth&nbsp;<span style="color:#ff3399">for</span>&nbsp;each&nbsp;point</div><div style="padding:0 6px; white-space:pre; line-height:130%">ground_mask:&nbsp;detected&nbsp;ground&nbsp;<span style="color:#ff3399">or</span>&nbsp;<span style="color:#ff3399">not</span>&nbsp;<span style="color:#ff3399">for</span>&nbsp;each&nbsp;point</div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%">&nbsp;</div><div style="padding:0 6px; white-space:pre; line-height:130%">points_cam&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;unproject_to_pointcloud(depth,&nbsp;fov)</div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%">points&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;points_cam</div><div style="padding:0 6px; white-space:pre; line-height:130%">ground_mask&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;ground_mask.flatten()</div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%">canonicalized&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;<span style="color:#4be6fa">False</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;</div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%"><span style="color:#ff3399">if</span>&nbsp;ground_mask.mean()&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">&gt;</span>&nbsp;canonicalize_threshold:</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;canonicalized&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;<span style="color:#4be6fa">True</span></div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;ground_pcd&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;subset(points_cam,&nbsp;ground_mask)</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;plane,&nbsp;_&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;ground_pcd.segment_plane(</div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;distance_threshold<span style="color:#0086b3"></span><span style="color:#ff3399">=</span><span style="color:#c10aff">0.</span><span style="color:#c10aff">05</span>,&nbsp;ransac_n<span style="color:#0086b3"></span><span style="color:#ff3399">=</span><span style="color:#c10aff">3</span>,&nbsp;num_iterations<span style="color:#0086b3"></span><span style="color:#ff3399">=</span><span style="color:#c10aff">1000</span>)</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;<span style="color:#ff3399">if</span>&nbsp;array([<span style="color:#c10aff">0</span>,&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">-</span><span style="color:#c10aff">1</span>,&nbsp;<span style="color:#c10aff">0</span>])&nbsp;@&nbsp;plane[:<span style="color:#c10aff">3</span>]&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">&lt;</span>&nbsp;<span style="color:#c10aff">0</span>:</div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;plane&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">-</span>plane</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;a,&nbsp;b,&nbsp;c,&nbsp;d&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;plane</div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;normal&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;array([a,&nbsp;b,&nbsp;c])</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;ez&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;array([<span style="color:#c10aff">0</span>,&nbsp;<span style="color:#c10aff">0</span>,&nbsp;<span style="color:#c10aff">1</span>])</div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;new_y&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;ez&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">-</span>&nbsp;normal&nbsp;@&nbsp;ez&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">*</span>&nbsp;normal</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;new_y&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;new_y&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">/</span>&nbsp;norm(new_y)</div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;rot&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;array([cross_prod(new_y,&nbsp;normal),&nbsp;new_y,&nbsp;normal])</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;rot&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;array([[<span style="color:#c10aff">0</span>,&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">-</span><span style="color:#c10aff">1</span>,&nbsp;<span style="color:#c10aff">0</span>],&nbsp;[<span style="color:#c10aff">1</span>,&nbsp;<span style="color:#c10aff">0</span>,&nbsp;<span style="color:#c10aff">0</span>],&nbsp;[<span style="color:#c10aff">0</span>,&nbsp;<span style="color:#c10aff">0</span>,&nbsp;<span style="color:#c10aff">1</span>]])&nbsp;@&nbsp;rot</div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;trans&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;array([<span style="color:#c10aff">0</span>,&nbsp;<span style="color:#c10aff">0</span>,&nbsp;d])</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;points_world&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;points_cam&nbsp;@&nbsp;rot.T&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">+</span>&nbsp;trans[<span style="color:#4be6fa">None</span>]</div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;points&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;points_world</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;</div><div style="background-color:#303030; padding:0 6px; white-space:pre; line-height:130%"><span style="color:#ff3399">return</span>&nbsp;points,&nbsp;canonicalized</div></div><div style="text-align:right;margin-top:-13px;margin-right:5px;font-size:9px;font-style:italic"><a href="http://colorscripter.com/info#e" target="_blank" style="color:#4f4f4ftext-decoration:none">Colored by Color Scripter</a></div></td><td style="vertical-align:bottom;padding:0 2px 4px 0"><a href="http://colorscripter.com/info#e" target="_blank" style="text-decoration:none;color:white"><span style="font-size:9px;word-break:normal;background-color:#4f4f4f;color:white;border-radius:10px;padding:1px">cs</span></a></td></tr></table></div>
</details>

<br>

- **(c) Lifting 2D Contexts to 3D Contexts**

&nbsp;&nbsp;&nbsp;&nbsp;기존 object detection이나 bounding box positioning을 사용하여 얻어진 VQA dataset은 2D(and pixel-level)의 reasoning 수준에 머물러 있습니다. 그 이유는 depth, altitude, distance등의 spatial information이 포함된 context가 부족하기 떄문입니다. 따라서 저자들은 이러한 문제점을 해결하기 위해, 우선 2D pixel을 metric-scale의 3D point cloud로 lift 해주었습니다(by depth estimation). 그 이후에는, 위에서 잠깐 언급한 바와 같이, segmentation model을 사용하여 point cloud들을 camera coordinates에서 world coordinate로 projection해주었다고 합니다.

<br>

- **(d) Ambiguity Resolution**