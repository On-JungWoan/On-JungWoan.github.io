---
title:  "[CVPR 2024] InterHandGen 논문 리뷰"
excerpt: "InterHandGen: Two-Hand Interaction Generation via Cascaded Reverse Diffusion"

categories:
  - papers
tags:
  - "DL"
  - "computer_vision"
  - "human-motion"

published: true

toc: true
toc_sticky: true
 
date: 2024-05-09
last_modified_at: 2024-05-09
use_math: true
permalink: "/dl_paper/interhandgen/"
legacy_category: DL_paper
---

> 논문링크 : <a href="https://jyunlee.github.io/projects/interhandgen/data/interhandgen.pdf" target="blank_">InterHandGen: Two-Hand Interaction Generation via Cascaded Reverse Diffusion</a>

> Implementation : <a href="https://github.com/jyunlee/InterHandGen" target="blank_">https://github.com/jyunlee/InterHandGen</a>

오늘 소개드릴 페이퍼는 CVPR 2024에 accept된 `InterHandGen: Two-Hand Interaction Generation via Cascaded Reverse Diffusion`입니다.

<br>
<br>

## 1. Abstract & Introduction

&nbsp;&nbsp;&nbsp;&nbsp;본 페이퍼에서는, MANO parameter를 generate하는 diffusion 모델을 학습하고, 해당 모델을 prior로 사용하여 다른 regression 모델들을 학습하는 방법에 대해 소개하고 있습니다. 따라서 해당 paper는 크게 `train diffusion`, `inference`의 두가지 stage로 나눌 수 있습니다.

&nbsp;&nbsp;&nbsp;&nbsp;`train diffusion` stage에서는 two hand interaction에 대한 MANO parameter를 generation할 수 있도록 diffusion 모델을 학습합니다. 이 때, 왼손과 오른손에 대한 파라미터를 한꺼번에 generation할 경우, train complexity가 매우 높아지기 때문에 비효율적입니다. 따라서 저자는 이를 해결하기 위해 `left hand와 right hand를 따로 generation` 합니다. 또한, left hand와 right hand가 대칭이라는 점을 활용하여, $p_{\phi}(x_l)$과 $p_{\phi}(x_r \| x_l)$을 jointly하게 학습합니다. 다시말해, `left hand의 shape을 먼저 generate한 다음 해당 shape을 right hand의 condition으로 사용`함으로써 generation process에 사용되는 DoF를 획기적으로 줄일 수 있었다고 합니다.

<p align="center"><img width="50%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/b0300233-a449-4e18-9d70-14474ba2c534"></p>

&nbsp;&nbsp;&nbsp;&nbsp;이후 `inference` stage에서는 diffusion model을 freeze한 뒤, 해당 모델을 `diffusion prior`로 사용하여 새로운 regression 모델을 학습합니다. 해당 prior는 regression 모델 학습 이외에도 optimization 방법이나 regularization term 등 다양한 곳에 활용될 수 있다고 합니다. 자세한 내용은 Method 파트에서 설명드리도록 하겠습니다.

<br>
<br>

## 2. Method

### 2-1. Problem Definition and Key Formulation

&nbsp;&nbsp;&nbsp;&nbsp;학습 과정에 대해 설명드리기에 앞서 Key formulation에 대해 먼저 언급하고 넘어가도록 하겠습니다. 우선, diffusion model은 input으로 들어온 noised MANO parameter에 대해 denoising을 진행하여 최종적으로 `denoised MANO parameter를 generation`합니다. 해당 MANO parameter는 다음과 같이 표현되며, 각각 MANO pose&shape parameter, root rotation(6D representation), root translation을 의미합니다.

<p align="center"><img width="35%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/62f95675-0459-4992-8802-afccffaebf9d"></p>

<br>

&nbsp;&nbsp;&nbsp;&nbsp;Abstract 부분에서도 언급했듯이, training complexity 문제로 인해 해당 diffusion 모델은 left hand와 right hand를 따로따로 generate합니다. 이 때, 양 손이 대칭이라는 insight를 활용하여 다음과 같이 `conditional & unconditional distribution을 jointly하게 학습`합니다.

<p align="center"><img width="50%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/2116240a-5630-441b-bd6d-5ea07487a263"></p>

<br>

&nbsp;&nbsp;&nbsp;&nbsp;하지만, 해당 방법은 2개의 hand space(left, right)에 대한 distribution을 학습해야하는데, 해당 embedding space 상에는 개념적으로 불필요한 중복이 존재합니다. 각 distribution은 결국 그럴싸한 single hand space를 capture한다는 궁극적인 공통점이 있고, 차이점이라고는 왼손인지 오른손인지, condition인지 아닌지가 전부입니다. 따라서 저자들은 이러한 불필요한 중복을 제거하기 위해 `mirroring transformation을 도입`($\Gamma$) 합니다.

&nbsp;&nbsp;&nbsp;&nbsp;이를 설명하기 위해서는 먼저 MANO의 내부적인 연산에 대해 이해해야 합니다. MANO는 양 손에 대한 mesh를 rendering하는 것처럼 보이지만, 사실 내부적으로는 `right-hand space에 대한 단 하나의 hand model만 존재`합니다. left hand의 경우는 위에서 설명드린 mirroring transformation $\Gamma$를 사용하여 `right hand로 flip` 시켜준 뒤 mesh를 rendering하게 됩니다. 본 논문의 저자도 MANO의 이러한 특성을 이용하여 train process를 다음과 같이 단순화 하였습니다.

$$
\begin{matrix}
p_{\phi}(x_l, x_r) &=& p_{\phi}(x_l) p_{\phi}(x_r | x_l) \\
       &=& p_{\phi}(  \Gamma (x_l) \; \color{Red}{or} \; x_r  ) \; p_{\phi}(x_r | x_l \; \color{Red}{or} \; \Gamma (x_l)  | \Gamma (x_r) ) \\
       &=& p_{\phi}(x_r) p_{\phi}(x_r | x_l)
\end{matrix}
$$

<br>

&nbsp;&nbsp;&nbsp;&nbsp;이로써 training objective를 `right-hand 단 하나의 space`에 대한 conditional & unconditional distribution으로 줄일 수 있게 되었습니다. 또한, 이는 `Classifier-Free Guidance(CFG)`의 format과도 같기 때문에 linear interpolation을 통해 다음과 같이 표현될 수 있습니다. 이 때, $z_t$는 right hand를, $c$는 left hand를 의미함을 알 수 있습니다. 해당 내용이 이해가 되지 않으신 분들은 [CLASSIFIER-FREE DIFFUSION GUIDANCE](https://arxiv.org/pdf/2207.12598) paper를 읽고 오시길 추천드립니다.

<p align="center"><img width="70%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/5223ddf0-a5ad-434a-ad92-545d9a87fa36"></p>

<br>

### 2-2. Network Architecture

&nbsp;&nbsp;&nbsp;&nbsp;학습 알고리즘 설명에 앞서 이해를 돕기 위해 네트워크 아키텍처에 대해 먼저 소개하도록 하겠습니다. input으로는 Diffused Hand, Conditional Hand, Diffusion time, Conditional Object(해당 condition은 2-7에서 자세히 소개하겠습니다)을 받아 최종적으로 Denoised right hand를 generate합니다. 각 Embeding layer는 2개의 Linear Layer로 이루어져 있으며, input embedding간의 attention을 계산하기 위해 `Transformer encoder`를 사용합니다. 이는 다시 최종적으로 MLP의 input으로 들어가 `MANO Parameter를 generate`하게 됩니다.

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/a7559984-bf76-4f22-94a6-d2be21ef5f16" style="border: solid black 1px"></p>

&nbsp;&nbsp;&nbsp;&nbsp;2-1에서도 잠시 언급드렸듯이, 해당 diffusion model은 right hand만을 generation하며, left hand는 이를 flip하여 generation합니다. left hand를 generate할 때는 zero matrix를 condition MLP의 input으로 넣어주며, diffusion time은 positional encoding을 해줍니다. 이후, 생성된 left hand는 다시 Cond. Hand MLP의 input으로 들어가 right hand를 generate합니다.

<br>

### 2-3. Train

&nbsp;&nbsp;&nbsp;&nbsp;Forward process의 알고리즘은 다음과 같습니다. 앞에서 쭉 언급한 내용을 pseudo code로 작성한 내용이라 어렵지 않게 이해할 수 있습니다. 해당 내용에 대해 간단히 설명드리도록 하겠습니다.

&nbsp;&nbsp;&nbsp;&nbsp;먼저, $q(x_r, x_l)$또는 $q(\Gamma(x_l), \Gamma(x_r))$로부터 $(x_r, x_l)$ pair를 뽑아주고, 이를 2-2에서 설명드린 Diffusion model의 input으로 각각 넣어줍니다. 이 때, $p_{uncond}$에 따라 특정 확률로 conditional hand($x_l$)를 zero matrix로 만들어줍니다. 그 이후의 과정은 diffusion training process와 동일합니다.

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/ea66de6f-0816-4337-a0a5-c55b913ee1ee"></p>

<br>

### 2-4. Inference

&nbsp;&nbsp;&nbsp;&nbsp;inference 역시 지금까지 설명한 내용을 pseudo code로 작성한 것입니다. 각 단계에 대해 간단히 설명드리도록 하겠습니다.

&nbsp;&nbsp;&nbsp;&nbsp;먼저 left hand(anchor hand)를 generate 해줍니다. condition은 0로 두고 noised parameter와 time step t를 input으로 줘서 backward process를 진행하면 denoised parameter($x_0$)를 얻을 수 있습니다. 그리고 이를 `flip`함으로써 left hand mesh를 얻을 수 있게 됩니다.

&nbsp;&nbsp;&nbsp;&nbsp;이렇게 얻은 `left hand mesh는 right hand의 condition`으로 들어가게 됩니다. diffusion model의 condition에 관해서는 CFG의 방식을 따르기 때문에, 각 time step t에서의 noise값은 $w_{cfg}$에 대한 linear interpolation으로 표현됩니다. 그리고, 생성된 mesh(right hand)가 기존의 mesh(left hand)를 뚫고 들어가는 penetration 현상을 방지하기 위해 `Anti-penetration guidance`도 추가해줍니다. 해당 guidance에 대해서는 바로 다음 파트에서 자세히 설명드리도록 하겠습니다.

&nbsp;&nbsp;&nbsp;&nbsp;이와 같은 일련의 backward process가 끝나면 최종적으로 two-hand interaction에 대한 hand mesh를 generate 할 수 있게 됩니다.

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/fa2cd07f-6ece-4f22-897a-fa4dc9fc9cf4"></p>

<br>

### 2-5. Anti-penetration guidance

&nbsp;&nbsp;&nbsp;&nbsp;위에서도 잠시 언급드렸듯이, inference time에서 right hand(add hand)가 generate될 때, 기존 left hand(anchor hand)의 mesh를 뚫고 들어가지 못하도록 하는 Anti-penetration guidance가 사용되었습니다. 아래는 해당 guidance에 대한 수식입니다. i는 penetration이 일어난 right hand의 point를 의미하며 j는 해당 point로부터 가장 가까이에 있는 left hand의 point를 의미합니다. 이후, 해당 point에 l2 loss를 걸어줌으로써 `penetration이 일어난 right hand의 point가 left hand의 point cloud 밖으로 나올 수 있도록 도와줍니다.`

<p align="center"><img width="70%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/4269f938-4c13-4011-ba9b-0077074e2981"></p>

<br>

### 2-6. Generative Prior for Two-Hand Problems

&nbsp;&nbsp;&nbsp;&nbsp;해당 diffusion model은, 다른 regression model을 사용할 때 `diffusion prior`로 사용될 수도 있습니다. input image로부터 ($x_l, x_r$) pair를 reconstruction하는 regression model이 있다고 가정했을 때, diffusion model $D_{\phi}$이 `frozen critic`의 역할을 해줄 수 있다고 합니다.

&nbsp;&nbsp;&nbsp;&nbsp;이는 아래와 같은 수식으로 표현될 수 있습니다. $S(\cdot, \cdot, \cdot)$은 input으로 regression output$(x_l, x_r)$을 받아, 이를 diffusion model에 넣어 denoised interaction $(\hat x_l, \hat x_r)$을 generate하는 function입니다. 따라서, 해당 term은 regression model의 output과, 해당 output에 대한 denoised interaction의 l2 loss로 계산될 수 있습니다. 다시말해, 해당 term은 regression model의 output이 diffusion으로 학습한 `prior를 따라갈 수 있도록 도와주는 regularization term`이라고 할 수 있습니다. 해당 regularization term은 loss function 외에도 shape optimization 등 다양한 학습 방식의 prior로 사용될 수 있다고 합니다. 이에 대한 실험 결과는 experiments에서 더 자세히 설명드리도록 하겠습니다.

$$
\begin{matrix}

\mathcal{L}_{reg} &=& || S(D_{\phi}, x_l, x_r) - (x_l, x_r) ||_2 \\
                  &=& || (\hat x_l, \hat x_r) - (x_l, x_r) ||_2

\end{matrix}
$$

<br>

### 2-7. Object-conditional generation

&nbsp;&nbsp;&nbsp;&nbsp;2-2에서도 잠깐 언급했듯이, 본 페이퍼의 diffusion model은 two-hand interaction 이외에도 Hand-Obj Interaction(이하 HOI) 또한 generation할 수 있습니다. 이에 대한 수식은 다음과 같습니다. 아래 수식에서도 확인할 수 있듯이, `object에 대한 condition`만 간단히 추가함으로써 HOI generation을 할 수 있었다고 합니다. 이 때, conditional object(point cloud)는 PointNet-based embedding network의 input으로 들어가서 다른 embedding vector들과 함께 transformer의 input으로 들어가게 됩니다. 이후의 모든 과정은 앞서 설명드린 바와 같습니다.

$$
p_{\phi}(x_l, x_r | c) = p_{\phi}(x_l | c) p_{\phi}(x_r | x_l, c)
$$

<br>
<br>

## 3. Experiments

### 3-1. Generation

- **Two-hand interactions**

  &nbsp;&nbsp;&nbsp;&nbsp;`InterHand2.6m dataset`에 대해 generate한 결과입니다. plausible & diverse한 hand mesh들이 잘 생성된 것을 확인할 수 있습니다. `Anti-penetration guidance`의 영향으로 penetration 현상이 상당히 완화된 것을 확인할 수 있습니다.

  <p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/91563e94-32a6-4173-86d9-2cdd8c6b121b" style="border: solid black 1px"></p>

<br>

- **Object-Conditioned Two-Hand Synthesis**

  &nbsp;&nbsp;&nbsp;&nbsp;HOI generation 결과입니다. 마찬가지로 그럴싸한 hand mesh들이 생성된 것을 확인할 수 있습니다. Object point cloud에 대해서는 GT를 사용하는 것으로 보입니다. GT object point cloud가 condition으로 주어지면 그에 해당하는 Two-hand를 그럴싸하게 generate하는 것 같습니다. 해당 부분은 아직 코드에 공개된 사항이 없어서 추후 문의를 해볼 예정입니다.

  <p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/59fc9932-b693-40d0-a315-7996d1515129" style="border: solid black 1px"></p>


<br>

### 3-2. Monocular Two-Hand Reconstruction

&nbsp;&nbsp;&nbsp;&nbsp;InterhandGen을 diffusion prior로 사용했을 때의 퍼포먼스 향상에 대해 보여주고 있는 실험결과입니다. 동일 모델에 대해 해당 prior를 적용시켰을 때 `error가 감소`하는 것을 확인할 수 있었으며, visualization 결과 역시 눈에 띄게 좋아진 것을 확인할 수 있었다고 합니다.

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/85790135-88d3-49e7-97c5-0d1a6eb9f5ee" style="border: solid black 1px"></p>

<p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/80615af1-5e9b-454c-a515-63932b935f7a" style="border: solid black 1px"></p>

<br>

### 3-3. Ablation study

- **SlefAtt & CFG**

  &nbsp;&nbsp;&nbsp;&nbsp;`Classifier-free guidance(CFG) 및 input embedding`에 대해 적용했던 self-attention에 대한 ablation study 결과입니다.

  <p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/5a4a951c-d8b9-4a86-81fa-bbada536ed6a" style="border: solid black 1px"></p>

- **Anti-pen. guidance**

  &nbsp;&nbsp;&nbsp;&nbsp;`Anti-penetration guidance`에 대한 ablation study 결과입니다.

  <p align="center"><img width="100%" src="https://github.com/On-JungWoan/On-jungWoan/assets/84084372/a89c4bc4-445a-4334-bad5-1ea8ef8e4289" style="border: solid black 1px"></p>
