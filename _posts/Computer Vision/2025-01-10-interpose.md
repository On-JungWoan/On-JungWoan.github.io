---
title:  "[Arxiv 2024] Inter-pose 요약"
excerpt: "Can Generative Video Models Help Pose Estimation? (Inter-pose)"

categories:
  - DL_paper
tags:
  - [DL, computer_vision]

published: true

toc: true
toc_sticky: true
 
date: 2025-01-10
last_modified_at: 2025-01-10
use_math: true
---

**[Title]** Can Generative Video Models Help Pose Estimation?

**[Keyword]** Camera pose estimation, Generation

**[Journal]** Arxiv, 2024

**[arXiv]** <a href="https://arxiv.org/abs/2412.16155" target="blank_">https://arxiv.org/abs/2412.16155</a>

**[Summary]**

<p align="center"><video width="100%" src="https://inter-pose.github.io/static/videos/teaser.mp4" autoplay controls loop muted></video></p>

- 풀고자 하는 것?
  - corresponding feature가 전혀 없는 두 개의 frame의 global camera pose 찾기
- how?
  - 두 이미지 사이를 video generator로 채우고, 생성된 video를 SOTA camera estimator에 넣어서 camera pose estimate

<br>
<br>

## Main Idea

- 겹치는 부분이 전혀 없는 두 이미지 pair로부터 절대적인 camera pose를 찾는 것은 매우 어려움
  - why? 두 이미지로부터 어떠한 correspondence도 찾을 수 없기 때문에
    - COLMAP이나 다른 traditional approach들은 image pair 사이의 matching feature를 찾아서 camera pose를 estimate함. → matching feature에 매우 의존적.
    - 따라서 viewpoint가 갑작스럽게 변하는 부분(e.g. scene change)에 대해서는 camera pose 전혀 estimate x

- 이를 해결하기 위해 video generator가 가지고 있는 강력한 prior를 사용
  - video generator가 두 이미지 pair 사이의 frame을 상당히 그럴싸하게 채워줌.
  - 이후의 과정은 그냥 단순한 camera pose estimation 문제가 됨. → SOTA model로 그냥 단순히 camera 찾기만 하면 됨

- but, generator가 부자연스러운 영상을 생성하는 경우도 있음. (그냥 단순히 두 frame을 morphing해버림)
  - 따라서 두 이미지 pair에 대해 여러개의 generated video를 생성하고, 이 중 가장 잘 나온 video를 골라야 함.
  - but, 뭐가 잘 나온 video인지 어떻게 알아? → self-consistency score 도입
    - 여러개의 generated video에 대해 random frame index를 뽑고, 해당 frame을 사용하여 camera pose estimation 진행
    - 두 frame이 부드럽게 연결된 video는 어떤 frame을 선택하더라도 cam의 pose가 clurster 되어있을 것