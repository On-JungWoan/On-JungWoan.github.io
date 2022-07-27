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
 
date: 2022-07-27
last_modified_at: 2022-07-27
---

<br>

해당 데이터 셋의 크기가 큰 관계로, colab에서 작업시 리소스 부족의 우려가 있다. 
따라서 로컬 환경에서 개발을 진행하여준다. 
데이터셋 특성상, 학습 시간이 매우 느리므로 병렬 연산이 가능한 GPU를 사용하는 것이 좋다.
로컬 환경에서 GPU를 사용하기 위해 필요한 것이 CUDA와 cuDNN이다. 

<br>

## 0. CUDA, cuDNN 설치
### 0-0. GPU 사용 가능 여부 확인

다음 두 줄의 코드를 입력하여 본다.

```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```
```
[name: "/device:CPU:0"
 device_type: "CPU"
 memory_limit: 268435456
 locality {
 }
 incarnation: 12364090281161528455
 xla_global_id: -1]
 ```
나는 CPU에 관한 정보만 나왔으나, GPU에 관한 정보까지 나왔다면, GPU를 사용할 수 있다는 것이니, 아래 절차에 따라 GPU를 사용한다.

<br>

### 0-1. CUDA, cuDNN 설치
CUDA 11.3 -> cuDNN 8.2 -> pytorch 설치 순으로 진행하였다. <br>

> [CUDA 설치](https://developer.nvidia.com/cuda-downloads) <br>
> [cuDNN 설치](https://developer.nvidia.com/rdp/cudnn-download#a-collapse714-92)

<br>

### 0-2. 경로 이동
- 앞서 CUDA 파일이 있는 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6 경로에 다시 갑니다.
- 그리고 cuDNN 폴더 안의 모든 파일을 복사하의 위의 위치에 붙여넣는다.

![image](https://user-images.githubusercontent.com/84084372/181012376-17475437-a86c-4014-91d1-c71f15a2d8a5.png)

<br>

### 0-3. 환경변수 등록
환경변수 path에 다음과 같이 bin, include, lib 세가지의 path를 추가

![image](https://user-images.githubusercontent.com/84084372/181015230-701eb275-df8f-4936-bed1-fc62c1e6aa5d.png)

<br>


## 1. 데이터 분할
### 1-1. 데이터 구조
![image](https://user-images.githubusercontent.com/84084372/181020674-5dd947aa-1bae-4cdd-a902-4a054fa5f397.png)
![image](https://user-images.githubusercontent.com/84084372/181020709-ad96ae31-3ad5-467a-8fdf-38ffc6779fd7.png)

원본 데이터셋은 분류 클래스가 폴더로 구분되어 있으며 Train, Test, Validation 데이터가 구분되어있지 않다. 
각 폴더 안에는 해당 분류 클래스에 속하는 이미지 데이터들이 저장되어 있음. 
예를 들어, 질병이 없는 Cherry의 사진 데이터는 Cherry_healthy 클래스에 해당한다. 
정확한 학습 설계를 위해서는 학습에 앞서 이 데이터를 Train, Test, Validation 데이터로 나누어주고, 가각에 클래스에 해당하는 폴더에 저장하는 작업을 시행해야 한다.

<br>


### 1-2. 데이터 분할을 위한 폴더 생성
```python
import os
import shutil

original_dataset_dir = 'C:/Users/user/Downloads/dataset'   
classes_list = os.listdir(original_dataset_dir) 
 
base_dir = 'C:/Users/user/Downloads/splitted' 
# os.mkdir(base_dir)                                1회만 실행
 
train_dir = os.path.join(base_dir, 'train') 
# os.mkdir(train_dir)                               1회만 실행
validation_dir = os.path.join(base_dir, 'val')
# os.mkdir(validation_dir)                          1회만 실행
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)                                1회만 실행

for cls in classes_list:     
    # os.mkdir(os.path.join(train_dir, cls))        1회만 실행
    # os.mkdir(os.path.join(validation_dir, cls))   1회만 실행
    # os.mkdir(os.path.join(test_dir, cls))         1회만 실행
```

![image](https://user-images.githubusercontent.com/84084372/181021496-1b81094d-f5e5-4fda-8df7-6c1bbc1c449a.png)

train, test, validation 데이터가 각 클래스에 맞게 생성된 모습.

![image](https://user-images.githubusercontent.com/84084372/181021548-b375c6af-0634-4f49-8915-8339c4fe6fd2.png)

<br>

### 1-3. 데이터 분할과 클래스별 데이터 수 확인
```python
import math
 
for cls in classes_list:
    path = os.path.join(original_dataset_dir, cls) # dataset/~
    fnames = os.listdir(path) # dataset/~/~
 
    train_size = math.floor(len(fnames) * 0.6)
    validation_size = math.floor(len(fnames) * 0.2)
    test_size = math.floor(len(fnames) * 0.2)
    
    # Train 데이터
    train_fnames = fnames[:train_size]
    print("Train size(",cls,"): ", len(train_fnames))
    for fname in train_fnames: 
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(train_dir, cls), fname)
        shutil.copyfile(src, dst) # src(dataset/~/~)안에 있는 파일 내용을 dst(train 데이터 폴더) 경로로 복사

    # Validation 데이터
    validation_fnames = fnames[train_size:(validation_size + train_size)]
    print("Validation size(",cls,"): ", len(validation_fnames))
    for fname in validation_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(validation_dir, cls), fname)
        shutil.copyfile(src, dst)
        
    test_fnames = fnames[(train_size+validation_size):(validation_size + train_size +test_size)]

    # Test 데이터
    print("Test size(",cls,"): ", len(test_fnames))
    for fname in test_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(test_dir, cls), fname)
        shutil.copyfile(src, dst)
```
```
Train size( Apple___Apple_scab ):  378
Validation size( Apple___Apple_scab ):  126
Test size( Apple___Apple_scab ):  126
Train size( Apple___Black_rot ):  372
Validation size( Apple___Black_rot ):  124
Test size( Apple___Black_rot ):  124
Train size( Apple___Cedar_apple_rust ):  165
Validation size( Apple___Cedar_apple_rust ):  55
Test size( Apple___Cedar_apple_rust ):  55
Train size( Apple___healthy ):  987
Validation size( Apple___healthy ):  329
Test size( Apple___healthy ):  329
Train size( Cherry___healthy ):  512
Validation size( Cherry___healthy ):  170
.
.
.
(생략)
```
