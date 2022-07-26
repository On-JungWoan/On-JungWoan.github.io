---
title:  "[딥러닝] 작물 잎 사진으로 질병 분류하기 예제 풀이"
excerpt: "교재 : 파이토치 딥러닝 프로젝트 문제집"

categories:
  - DL
tags:
  - [딥러닝, CNN, 전이학습]

published: true

toc: true
toc_sticky: true
 
date: 2022-07-26
last_modified_at: 2022-07-26
---

## 0. CUDA, cuDNN 설치
데이터 셋의 크기가 큰 관계로, colab에서 작업시 리소스 부족의 우려가 있다. 
따라서 로컬 환경에서 GPU를 사용하여 개발을 진행하여준다. 
로컬 환경에서 GPU를 사용하기 위해 필요한 것이 CUDA와 cuDNN이다. 

<br>

### 0-1. CUDA, cuDNN 설치
> [CUDA 설치](https://developer.nvidia.com/cuda-downloads) <br>
> [cuDNN 설치](https://developer.nvidia.com/rdp/cudnn-download#a-collapse714-92)

### 0-2. 경로 이동
- 앞서 CUDA 파일이 있는 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6 경로에 다시 갑니다.
- 그리고 bin, include, lib 폴더 안에 cuDNN 에서 설치한 bin, include, lib 내용을 추가합니다.
