---
title:  "[ICML 2026] PhysHanDI 논문 리뷰"
excerpt: "PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions (ICML, 2026)"

categories:
  - DL_paper
tags:
  - [DL, computer_vision]

published: true

toc: true
toc_sticky: true

date: 2026-08-29
last_modified_at: 2026-08-29
use_math: true
---

**[Title]** PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions

**[Keyword]** Hand-Deformable Object Interaction, Spring-Mass Model, Inverse Physics

**[Journal]** ICML, 2026

**[Authors]** Jihyun Lee, Changmin Lee, Donghwan Kim, Tae-Kyun Kim

**[Project Page]** <a href="https://jyunlee.github.io/projects/physhandi/" target="_blank" rel="noopener noreferrer">https://jyunlee.github.io/projects/physhandi/</a>

**[arXiv]** <a href="https://arxiv.org/abs/2605.09538" target="_blank" rel="noopener noreferrer">https://arxiv.org/abs/2605.09538</a>

**[Summary]**

<p align="center"><img src="{{ '/assets/images/posts/physhandi/intro-examples.png' | relative_url }}" alt="천, 치마, 인형 등 여러 변형 물체와 손이 상호작용하는 PhysHanDI 입력 사례" width="100%"></p>

- Sparse-view RGB-D video에서 hand와 deformable object의 dense 3D motion을 함께 reconstruction함.
- Hand는 MANO model, object는 Spring-Mass Model로 표현하고, 두 모델을 virtual spring으로 연결해 손의 움직임을 물체에 가해지는 force로 변환함.
- Hand motion으로 object physics를 복원한 뒤, 복원된 physics를 inverse-physics prior로 사용해 hand pose를 다시 refinement함.

<br>

## 1. Motivation

기존의 Hand-Object Interaction 연구들은 대부분 rigid object나 articulated object를 대상으로 해왔습니다. 컵이나 통조림 같은 rigid object는 shape 자체가 변하지 않기 때문에, 각 frame에서 object의 rotation과 translation을 추정하면 전체 motion을 표현할 수 있습니다. 가위나 노트북처럼 articulated되는 object도 몇 개의 rigid part로 나눈 뒤, 각 part의 transformation이나 joint angle과 같은 비교적 적은 수의 parameter로 움직임을 표현할 수 있습니다.

<div style="display:flex;flex-wrap:wrap;justify-content:center;align-items:flex-start;gap:1rem;">
  <figure style="flex:1 1 280px;max-width:382px;margin:0;">
    <img src="{{ '/assets/images/posts/physhandi/rigid-object-examples.png' | relative_url }}" alt="형태가 변하지 않는 컵, 통조림, 병 형태의 rigid object 예시" width="100%">
    <figcaption style="display:block;width:100%;text-align:center;">Rigid objects: 컵, 통조림, 병</figcaption>
  </figure>
  <figure style="flex:1 1 280px;max-width:382px;margin:0;">
    <img src="{{ '/assets/images/posts/physhandi/articulated-object-examples.png' | relative_url }}" alt="여러 rigid part와 joint로 표현할 수 있는 가위와 노트북 형태의 articulated object 예시" width="100%">
    <figcaption style="display:block;width:100%;text-align:center;">Articulated objects: 가위, 노트북</figcaption>
  </figure>
</div>

하지만 천, 파우치, 인형과 같은 deformable object에서는 상황이 많이 달라집니다. 손으로 물체의 한 부분을 잡아당기거나 누르면, 물체의 각 부분이 서로 다른 방향과 크기로 움직이면서 shape 자체가 계속 변합니다. 따라서 하나의 rotation과 translation만으로는 전체 motion을 표현할 수 없으며, 훨씬 높은 degree of freedom을 갖는 deformation을 모델링해야 합니다.

이 deformation은 영상에서 보이는 geometry만의 문제도 아닙니다. 손이 어디를 잡고 어떻게 움직였는지, 물체가 얼마나 단단하거나 잘 늘어나는지와 같은 물리적 특성에 따라 결과가 달라집니다. 같은 hand motion이 주어지더라도 부드러운 천과 단단한 고무는 전혀 다른 형태로 변형될 수 있습니다.

PhysHanDI는 hand와 object를 각각 reconstruction하는 데서 끝나지 않습니다. 먼저 reconstruction된 hand motion으로 object에 가해지는 interaction force를 모델링하고, physics simulation을 통해 object의 deformation을 복원합니다. 그리고 여기서 한 단계 더 나아가, 복원한 object physics를 다시 사용해 hand pose까지 refinement하는 양방향 구조를 제안합니다.

<br>

## 2. Preliminary: Spring-Mass Model

PhysHanDI를 이해하기 위해서는 먼저 deformable object를 표현하는 **Spring-Mass Model**을 살펴볼 필요가 있습니다. 직관적으로는 하나의 물체를 여러 개의 작은 point로 나누고, 이 point들을 spring으로 연결한 구조라고 생각하면 됩니다. 외부에서 힘이 가해졌을 때 각 point가 어떻게 움직이는지를 물리 법칙에 따라 계산함으로써 전체 object의 motion과 deformation을 simulation합니다.

<p align="center"><img src="{{ '/assets/images/posts/physhandi/spring-mass-model.png' | relative_url }}" alt="물체를 mass node와 spring edge의 graph로 표현하는 Spring-Mass Model" width="100%"></p>

조금 더 구체적으로는 object를 graph $\mathcal{O}=(\mathcal{N},\mathcal{E})$로 표현합니다. $\mathcal{N}$은 object를 구성하는 mass node의 집합이고, $\mathcal{E}$는 node들을 연결하는 spring의 집합입니다. 각 node $\mathbf{n}\_i$는 position $\mathbf{x}\_i$, velocity $\mathbf{v}\_i$, mass $m\_i$를 가지며, 각 spring에는 stiffness $s\_{ij}$와 damping coefficient $\gamma\_{ij}$가 정의됩니다.

각 node에 작용하는 force는 크게 spring force, damping force, external force로 구성됩니다.

$$
\mathbf{F}_i = \sum_{(i,j)\in\mathcal{E}}
\left(\mathbf{F}_{i,j}^{\text{spring}} + \mathbf{F}_{i,j}^{\text{damping}}\right)
+ \mathbf{F}_i^{\text{external}}.
$$

### 2-1. Spring force

Spring force는 연결된 두 node 사이의 거리를 초기 상태의 거리인 rest length $r\_{ij}$로 되돌리는 방향으로 작용합니다.

$$
\mathbf{F}_{i,j}^{\text{spring}}
= s_{ij}\left(\lVert\mathbf{x}_j-\mathbf{x}_i\rVert-r_{ij}\right)
\frac{\mathbf{x}_j-\mathbf{x}_i}{\lVert\mathbf{x}_j-\mathbf{x}_i\rVert}.
$$

예를 들어 두 node 사이의 초기 거리가 10 cm였는데 이후 15 cm로 늘어났다면, spring force는 두 node의 거리를 다시 10 cm에 가깝게 줄이는 방향으로 작용합니다. 이때 $s\_{ij}$는 spring의 stiffness를 나타냅니다. 값이 클수록 더 강한 restoring force가 발생하므로 물체가 단단하게 움직이고, 값이 작을수록 쉽게 늘어나거나 휘어집니다.

### 2-2. Damping force

Damping force는 두 node 사이의 상대적인 움직임을 줄여주는 힘입니다.

$$
\mathbf{F}_{i,j}^{\text{damping}}=-\gamma_{ij}(\mathbf{v}_i-\mathbf{v}_j).
$$

Spring force만 사용하면 한 번 변형된 물체가 원래 상태로 돌아오는 과정에서 계속 진동할 수 있습니다. Damping force는 이러한 진동을 점차 줄여주며, $\gamma\_{ij}$는 움직임이 얼마나 빠르게 감쇠하는지를 결정합니다. $\gamma\_{ij}$가 작으면 더 탱탱하고 잘 튕기는 움직임을, 크면 더 점성 있고 묵직한 움직임을 표현할 수 있습니다.

<div style="display:flex;flex-wrap:wrap;justify-content:center;align-items:flex-start;gap:1rem;">
  <figure style="flex:1 1 280px;max-width:382px;margin:0;">
    <img src="{{ '/assets/images/posts/physhandi/damping-low.gif' | relative_url }}" alt="damping coefficient가 작은 spring system의 지속적인 진동" width="100%">
    <figcaption style="display:block;width:100%;text-align:center;">작은 damping coefficient: 진동이 오래 유지됨</figcaption>
  </figure>
  <figure style="flex:1 1 280px;max-width:368px;margin:0;">
    <img src="{{ '/assets/images/posts/physhandi/damping-high.gif' | relative_url }}" alt="damping coefficient가 큰 spring system의 빠른 진동 감쇠" width="100%">
    <figcaption style="display:block;width:100%;text-align:center;">큰 damping coefficient: 진동이 빠르게 감쇠함</figcaption>
  </figure>
</div>

마지막 external force는 중력과 같이 system 외부에서 작용하는 힘을 의미합니다. 이렇게 각 node에 작용하는 force를 계산한 뒤에는 Newton의 운동 법칙을 시간에 대해 적분해 다음 timestep의 velocity와 position을 업데이트합니다.

$$
\mathbf{v}_i^{t+1}=\mathbf{v}_i^t+\Delta t\frac{\mathbf{F}_i}{m_i},
\qquad
\mathbf{x}_i^{t+1}=\mathbf{x}_i^t+\Delta t\,\mathbf{v}_i^{t+1}.
$$

이 과정을 object의 모든 node에 반복하면 시간에 따른 deformation을 simulation할 수 있습니다. PhysHanDI는 실제 RGB-D observation과 simulation 결과가 가까워지도록 stiffness $s\_{ij}$, damping coefficient $\gamma\_{ij}$ 등의 physics parameter를 optimize합니다.

<br>

## 3. Hand-Deformable Object Interaction Modeling

Spring-Mass Model만으로는 object 자체의 deformation을 simulation할 수 있지만, 실제 Hand-Object Interaction을 표현하려면 손의 움직임을 object에 전달하는 방법이 추가로 필요합니다. PhysHanDI는 이를 **virtual spring**으로 해결합니다.

<p align="center"><img src="{{ '/assets/images/posts/physhandi/virtual-springs.png' | relative_url }}" alt="MANO hand vertex와 deformable object node 사이에 생성된 virtual spring" width="100%"></p>

먼저 각 MANO hand vertex와 object node 사이의 거리를 계산합니다. 두 점의 거리가 connection radius $\delta$보다 가까우면 서로 contact하고 있다고 판단하고, 해당 hand vertex와 object node 사이에 virtual spring을 추가합니다. 기존 Spring-Mass Model이 object 내부의 node들만 spring으로 연결했다면, 여기서는 hand와 object를 연결하는 spring을 하나 더 만드는 셈입니다.

Reconstruction된 MANO trajectory를 따라 hand가 움직이면 hand vertex의 위치도 함께 이동하고, 이에 따라 virtual spring의 길이와 상대 velocity가 변합니다. 그 결과 virtual spring에서 spring force와 damping force가 발생하고, 이 force가 contact된 object node로 전달됩니다. 전달된 force는 다시 object 내부의 spring을 통해 주변 node로 퍼지면서 전체 deformable object의 motion과 deformation을 만들어냅니다.

정리하면 virtual spring은 **hand motion을 object에 전달되는 physical force로 변환하는 연결고리**입니다. Sparse point controller를 사용하는 기존 방법과 달리 dense MANO mesh를 사용하기 때문에, 가려진 contact 영역까지 포함한 full 3D hand geometry를 interaction cue로 활용할 수 있다는 점이 중요합니다.

<br>

## 4. Learning from Sparse-View RGB-D Videos

<p align="center"><img src="{{ '/assets/images/posts/physhandi/method-overview.png' | relative_url }}" alt="Hand Reconstruction, Object Reconstruction, Hand Refinement로 이루어진 PhysHanDI 전체 파이프라인" width="100%"></p>

PhysHanDI는 sparse-view RGB-D video와 calibration된 camera intrinsic/extrinsic parameter를 입력으로 사용합니다. 전체 pipeline은 다음 세 단계로 나뉩니다.

1. **Hand Reconstruction**: Multi-view RGB-D observation에 MANO model을 fitting함.
2. **Object Reconstruction**: Reconstruction된 hand motion을 고정하고 Spring-Mass Model의 geometry와 physics parameter를 optimize함.
3. **Hand Refinement**: 학습한 object physics를 고정하고 inverse physics를 통해 hand pose를 다시 optimize함.

핵심은 2번과 3번 사이의 순환적인 관계입니다. Hand motion은 object physics를 학습하기 위한 prior가 되고, object physics는 다시 hand reconstruction을 개선하기 위한 prior가 됩니다.

### 4-1. Hand Reconstruction

첫 번째 단계는 일반적인 multi-view MANO fitting 과정과 비슷합니다. 각 frame에서 MANO pose, shape, global rotation, translation으로 이루어진 parameter $\Theta\_h$를 optimize합니다.

$$
\min_{\Theta_h}\;
\mathcal{L}_{2D}(\Theta_h,\mathbf{U})
+\lambda_d\mathcal{L}_d(\Theta_h,\mathbf{D})
+\lambda_t\mathcal{L}_t(\Theta_h,\Theta_h^{\text{prev}}).
$$

$\mathcal{L}\_{2D}$는 projected MANO keypoint와 2D keypoint supervision 사이의 reprojection error, $\mathcal{L}\_d$는 rendered MANO depth와 observed depth 사이의 차이, $\mathcal{L}\_t$는 이전 frame과의 temporal smoothness를 측정합니다.

이 단계 자체에 특별히 새로운 loss가 추가되는 것은 아닙니다. 목적은 multi-view 정보를 활용해 신뢰할 수 있는 dense 3D hand motion을 먼저 확보하고, 이후 이 motion을 object physics optimization의 condition으로 사용하는 것입니다.

### 4-2. Object Reconstruction

#### 4-2-1. Initial object geometry

Spring-Mass Model을 simulation하려면 시작점이 되는 object의 초기 3D geometry, scale, rotation, translation이 필요합니다. PhysHanDI는 첫 frame의 object image를 TRELLIS 계열의 image-to-3D model에 넣어 초기 mesh를 생성합니다.

다만 image-to-3D model이 생성한 mesh의 scale과 pose는 실제 RGB-D coordinate에 맞지 않습니다. 이를 해결하기 위해 mesh를 여러 방향에서 rendering하고, rendered mesh point와 실제 RGB image pixel 사이의 feature correspondence를 구합니다. Correspondence가 가장 많이 얻어지는 방향을 초기 rotation으로 선택한 뒤, matching 결과가 더 잘 맞도록 rotation을 refinement합니다. 이후 depth observation을 이용해 scale과 translation도 조정합니다.

<p align="center"><img src="{{ '/assets/images/posts/physhandi/object-initialization.png' | relative_url }}" alt="여러 방향으로 렌더링한 mesh와 RGB image의 feature를 matching해 object pose를 초기화하는 과정" width="100%"></p>

그래도 생성된 mesh와 실제 첫 frame의 deformable object shape 사이에는 차이가 남을 수 있습니다. 저자들은 As-Rigid-As-Possible(ARAP) shape optimization을 추가로 적용해 local geometry를 가능한 한 유지하면서 mesh를 depth observation에 맞게 부드럽게 deformation합니다. 이렇게 얻은 mesh가 Spring-Mass Model의 initial geometry가 됩니다.

#### 4-2-2. Inverse physics optimization

초기 geometry와 hand trajectory를 얻었다면, 다음은 object의 물성을 찾는 과정입니다. 먼저 reconstruction된 MANO trajectory를 고정하고 현재 physics parameter로 Spring-Mass Model을 simulation합니다. 이후 simulation된 object와 실제 RGB-D video에서 관측된 object motion을 비교합니다. 두 결과의 차이가 크면 stiffness, damping coefficient, connection radius 등의 parameter를 업데이트한 뒤 다시 simulation합니다.

<p align="center"><img src="{{ '/assets/images/posts/physhandi/object-reconstruction.png' | relative_url }}" alt="MANO hand motion을 고정하고 object model의 physics parameter를 inverse physics로 최적화하는 과정" width="100%"></p>

Supervision으로는 두 가지 정보를 사용합니다.

- $\mathcal{L}\_{ch}$: Depth map에서 lifting한 observed 3D point cloud와 simulated object node 사이의 Chamfer Distance
- $\mathcal{L}\_{tr}$: CoTracker3로 얻은 pseudo-ground-truth 3D point trajectory와 simulated node trajectory 사이의 $\ell\_2$ loss

결과적으로 이 과정은 실제 object가 움직이는 모습을 보고, **“이 물체는 어느 정도로 단단하고, 움직임이 어느 정도로 감쇠되는 물체인가?”**를 역으로 추정하는 system identification이라고 볼 수 있습니다. 최종적으로 주어진 hand motion에 대해 실제 observation과 유사한 deformation을 만들어내는 Spring-Mass Model을 얻습니다.

### 4-3. Hand Refinement

마지막 단계에서는 앞의 optimization 방향을 반대로 바꿉니다. Object Reconstruction에서는 hand pose를 고정하고 object physics를 optimize했다면, Hand Refinement에서는 학습한 Spring-Mass Model을 고정하고 hand pose를 optimize합니다.

<p align="center"><img src="{{ '/assets/images/posts/physhandi/hand-refinement.png' | relative_url }}" alt="복원된 object physics를 고정하고 MANO hand pose를 inverse physics로 refinement하는 과정" width="100%"></p>

예를 들어 손가락 일부가 object에 가려져 initial MANO pose에 작은 오차가 생겼다고 가정해보겠습니다. 잘못된 hand pose로 simulation하면 실제 observation과 다른 object deformation이 발생합니다. PhysHanDI는 simulation된 object가 실제 geometry와 motion에 가까워지도록 MANO parameter를 다시 optimize합니다.

이때도 Object Reconstruction과 동일한 $\mathcal{L}\_{ch}$와 $\mathcal{L}\_{tr}$를 사용하지만, gradient가 update하는 대상은 object physics가 아니라 MANO parameter입니다. 영상에서 직접 관측되는 hand cue뿐만 아니라, **“이 hand pose라면 실제로 이런 object deformation이 발생해야 한다”**는 physical consistency까지 활용하는 것입니다.

<br>

## 5. Experiments

저자들은 PhysTwin dataset 가운데 dense contact가 포함된 sequence를 **PhysTwin-dense**로 구분하고, 추가로 synchronized three-view RGB-D video 19개로 이루어진 **DenseHDI** dataset을 구축했습니다. 실험은 크게 fitting에 사용한 구간을 복원하는 Reconstruction & Resimulation과, 학습에 사용하지 않은 이후 frame을 예측하는 Future Prediction으로 나뉩니다. 비교 대상은 PhysTwin, Spring-Gaus, GS-Dynamics입니다.

<p align="center"><img src="{{ '/assets/images/posts/physhandi/experiments-qualitative.png' | relative_url }}" alt="PhysHanDI와 PhysTwin, Spring-Gaus, GS-Dynamics의 reconstruction과 future prediction 정성 비교" width="100%"></p>

Qualitative result에서는 PhysHanDI의 simulation이 실제 observation과 hand contact에 더 잘 정렬되는 것을 확인할 수 있습니다. Spring-Gaus는 sparse three-view setting에서 geometry가 불안정해지고, GS-Dynamics는 짧은 sequence의 큰 deformation을 충분히 표현하지 못하는 경향을 보입니다. PhysTwin은 sparse controller point를 사용하기 때문에 실제 contact 주변의 force를 정밀하게 전달하는 데 한계가 있습니다.

PhysTwin-dense의 Reconstruction & Resimulation 결과를 보면, PhysTwin 대비 dynamic-region Chamfer Distance는 10.78에서 **8.32**로, full-object Chamfer Distance는 5.90에서 **5.30**으로 감소했습니다. Tracking Error 역시 1.00에서 **0.89**로 줄었으며, 2D IoU와 PSNR도 개선되었습니다. Future Prediction에서도 대부분의 metric에서 같은 경향이 나타납니다.

<p align="center"><img src="{{ '/assets/images/posts/physhandi/experiments-quantitative.png' | relative_url }}" alt="PhysHanDI hand refinement의 single-view future prediction 결과와 입력 perturbation에 대한 robustness 분석" style="width:100%;max-width:585px;"></p>

특히 single-view future prediction에서 Hand Refinement의 효과가 잘 드러납니다. Refinement를 제거했을 때와 비교하면 object CD는 42.8에서 **33.5**, Track Error는 7.36에서 **6.75**, Hand CD는 7.57에서 **7.17**로 감소합니다. Object physics가 hand reconstruction을 개선하는 실제 prior로 동작한다는 것을 보여주는 결과입니다.

Noise에 대한 robustness도 흥미롭습니다. CoTracker signal에 perturbation을 주었을 때 PhysTwin-dense에서 PhysTwin의 CD는 clean input 대비 3.70 증가하지만, PhysHanDI는 0.26만 증가합니다. Dense MANO hand reconstruction이 sparse depth point보다 안정적인 contact cue를 제공한다는 저자들의 주장을 뒷받침합니다.

<br>

## 6. Applications

<p align="center"><img src="{{ '/assets/images/posts/physhandi/applications.gif' | relative_url }}" alt="PhysHanDI를 이용한 키보드 기반 실시간 object simulation과 robot teleoperation 응용" width="100%"></p>

개인적으로 이 paper에서 가장 재미있는 부분은 application입니다. 한 번 object의 Spring-Mass Model을 학습해두면, 새로운 hand motion을 controller로 사용해 object deformation을 real-time으로 simulation할 수 있습니다. 왼쪽 결과는 keyboard로 양손을 움직이며 인형의 deformation을 제어하는 예시이고, 오른쪽은 robot teleoperation에 적용한 결과입니다.

단순히 관측된 sequence를 3D로 다시 만드는 reconstruction을 넘어, 복원한 physics model을 새로운 interaction에 재사용할 수 있다는 점이 중요합니다. Human demonstration으로부터 deformable object dynamics를 학습하고 이를 robot control이나 simulation에 연결하는 방향으로 확장한다면 활용도가 높아 보입니다.

<br>

## 7. Discussion

PhysHanDI의 핵심 contribution은 다음과 같이 정리할 수 있습니다.

1. Sparse-view RGB-D video에서 dense 3D hand와 non-rigid object를 함께 reconstruction함.
2. Dense MANO motion과 Spring-Mass Model을 virtual spring으로 연결해 contact force를 simulation함.
3. Hand motion으로 object physics를 학습하고, 다시 object physics로 hand pose를 개선하는 cyclic inverse-physics pipeline을 제안함.
4. 학습한 physics model을 future prediction, real-time interaction, robot teleoperation에 재사용할 수 있음.

한편 현재 method는 sequence 안에서 hand-object contact topology가 유지된다고 가정합니다. 즉, interaction 도중 손을 떼거나 다른 위치를 다시 잡는 dynamic contact change를 직접 모델링하지는 않습니다. Virtual spring이 전달하는 force 역시 실제 finger pressure를 추정한 값이라기보다는 Spring-Mass simulation을 구동하는 boundary condition에 가깝습니다. 또한 입력으로 calibrated sparse-view RGB-D video가 필요하고, full pipeline에는 MANO fitting과 hand refinement의 추가 optimization cost가 들어갑니다.

그럼에도 deformable object reconstruction과 hand reconstruction을 서로 독립적인 문제로 보지 않고, **물체의 변형이 손의 움직임을 설명하고 손의 움직임이 다시 물체의 변형을 설명하도록 닫힌 loop를 만들었다**는 점이 인상적인 연구입니다. 특히 단순하고 해석 가능한 Spring-Mass Model을 dense hand reconstruction과 결합해 실제 future prediction과 interactive simulation까지 연결한 부분이 PhysHanDI의 가장 큰 장점이라고 생각합니다.

더 자세한 qualitative result와 supplementary video는 [PhysHanDI Project Page](https://jyunlee.github.io/projects/physhandi/)에서 확인할 수 있습니다.
