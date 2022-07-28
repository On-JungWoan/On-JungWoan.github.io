---
title:  "[딥러닝] 작물 잎 사진으로 질병 분류하기 예제 풀이 #1"
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

![image](https://user-images.githubusercontent.com/84084372/181021548-b375c6af-0634-4f49-8915-8339c4fe6fd2.png)

train, test, validation 데이터가 각 클래스에 맞게 생성된 모습.

## 2. 베이스라인 모델 학습
### 2-1. 베이스라인 모델 학습을 위한 준비

미리 사용할 디바이스를 입력시킨다. 
현재 내가 사용중인 로컬 pc는 gpu를 지원하지 않으므로, device에는 cpu가 입력된다. 
batch 사이즈와 epoch 수는 각각 256과 30으로 지정하였다. 

```python
import torch
import os
 
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

BATCH_SIZE = 256 
EPOCH = 30 
```

```python
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder 
 
transform_base = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()]) 
train_dataset = ImageFolder(root='C:/Users/user/Downloads/splitted/train', transform=transform_base) 
val_dataset = ImageFolder(root='C:/Users/user/Downloads/splitted/val', transform=transform_base)
```

```python
from torch.utils.data import DataLoader

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
```

<br>

**cf) 전처리 과정에 대한 정리**

**1. transforms 정의** 
> - torchvision.transforms
>   - 데이터 불러올 때, transform 옵션에 넣어줌.
> - transform 옵션에 넣는 방법
>   - transforms.ToTensor()와 같이 직접 넣어주는 방법
>   - transforms.Compose()에 원하는 내용 넣어서 한꺼번에 묶어서 넣는 방법

<br>

**2. 데이터셋 불러오기**
> - torchvision.datasets
>   - torch에 저장되어있는 데이터셋을 불러올 때
> - torchvision.datasets.ImageFolder
>   - 본인이 직접 만들거나 다운로드받은 데이터셋을 불러올 때
>   - 단, 일정한 구조를 만족해야 함 (하나의 폴더가 하나의 클래스에 대응)
>   - MNIST를 예를 들면 0~9의 폴더가 존재하고 각각 폴더 안에 해당 숫자에 대응하는 이미지가 존재 (폴더 이름이 레이블이 됨)

<br>

**3. 미니 배치 구성**
> - torch.utils.data.DataLoader
>   - 원하는 데이터와 batch_size를 넣어준다.
>   - 시계열 데이터가 아닐 경우, 딥러닝이 시간 정보는 학습하지 못하게 shuffle 옵션을 꼭 True로 해준다.
>   - num_workers는 데이터 프로세싱에 CPU 코어를 얼마나 할당할지에 대한 옵션으로, 적당한 값을 지정해줘야 모든 프로세스에서 최적의 성능을 보인다.
>   - 일반적으로 코어 개수의 절반정도 수치가 가장 무난한 것으로 알려져 있기 때문에, num_workers 값으로 4를 지정해줬다.

<br>

### 2-2. 데이터 확인
Train 데이터는 약 2만4천개, Validation 데이터는 약 8천개 정도의 관측값이 존재한다.
```python
print(len(train_dataset), len(val_dataset))
```
```
23989 7989
```

<br>

각각의 관측치는 이미지와 레이블로 구성되어 있다. 
이미지는 [3,64,64]의 RGB 3채널 컬러 이미지이며, Label은 classes_list(클래스명 리스트)의 인덱스 번호로 저장되어있다.
```python
print("0번 데이터의 0번 원소 : ",train_dataset[0][0].shape)
print("0번 데이터의 1번 원소 : ", train_dataset[0][1])
print("0번 데이터의 Label : ", classes_list[ train_dataset[0][1] ])
```
```
0번 데이터의 0번 원소 :  torch.Size([3, 64, 64])
0번 데이터의 1번 원소 :  0
0번 데이터의 Label :  Apple___Apple_scab
```

<br>

다음은 이미지를 시각화 한 것이다. 
matplotlib에서 RGB 이미지를 표현하기 위해서는 [width, height, channel]의 shape를 가지고 있어야 하는데, 
현재 torch 데이터의 shape는 [channel, width, height]이므로, permute()를 사용해 reshape 해준다. 
premute(1,2,0)의 의미는, [3,64,64]에서 1번 원소(64), 2번 원소(64), 3번 원소(3)의 순으로 shape를 재배치 하라는 것이다.
```python
import matplotlib.pyplot as plt

image = train_dataset[0][0].permute(1,2,0)
label = classes_list[ train_dataset[0][1] ]

plt.figure(figsize=(8,8))
plt.axis('off')

plt.imshow(image)
plt.title(label)

plt.show()
```
![image](https://user-images.githubusercontent.com/84084372/181293873-5f7909ae-798f-4f63-b3bc-829ee9896ee2.png)

<br>

이미지 256개가 하나의 batch이고, 총 94개의 batch가 존재한다.
```python
first_batch = train_loader.__iter__().__next__()

print("총 batch의 수 :",len(train_loader), end="\n\n")
print("첫 번째 batch의 shape :", first_batch[0].shape, end="\n\n")
print("첫 번째 batch의 label (중간생략) :",first_batch[1][:10])
```
```
총 batch의 수 : 94

첫 번째 batch의 shape : torch.Size([256, 3, 64, 64])

첫 번째 batch의 label (중간생략) : tensor([30, 12,  3,  0,  8, 30,  4, 32, 25, 11])
```

<br>

### 2-3. CNN 구조 정의
- 입력 데이터

<p align="center">[3, 64, 64]</p>

<br>

- Tensor의 가로와 세로의 크기

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/181292290-ffe1bbe3-25be-4957-982c-bb770688b5e8.png"></p>

<br>

- 최종 출력 클래스

<p align="center">[33]</p>

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/181302583-632957e3-a12f-4623-ac9a-68f7c2541ec9.png"></p>

<br>

![image](https://user-images.githubusercontent.com/84084372/181301539-a1bc9526-a3ba-4b1b-a6e2-025b79173be5.png)

![image](https://user-images.githubusercontent.com/84084372/181301565-73543f28-cce6-43b7-8ddb-93f898bfe584.png)

![image](https://user-images.githubusercontent.com/84084372/181301586-dac76d89-1d87-45c8-8dfd-f2dbc2036507.png)

<br>

### 2-4. CNN 구조 설계
앞서 정의한 CNN 구조를 바탕으로 베이스라인 모델을 설계한다. 

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
class Net(nn.Module): 
  
    def __init__(self): 
    
        super(Net, self).__init__() 

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) 
        self.pool = nn.MaxPool2d(2,2)  
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)  

        self.fc1 = nn.Linear(4096, 512) 
        self.fc2 = nn.Linear(512, 33) 
    
    def forward(self, x):  
    
        x = self.conv1(x)
        x = F.relu(x)  
        x = self.pool(x) 
        x = F.dropout(x, p=0.25, training=self.training) 

        x = self.conv2(x)
        x = F.relu(x) 
        x = self.pool(x) 
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv3(x) 
        x = F.relu(x) 
        x = self.pool(x) 
        x = F.dropout(x, p=0.25, training=self.training)

        x = x.view(-1, 4096)  
        x = self.fc1(x) 
        x = F.relu(x) 
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x) 

        return F.log_softmax(x, dim=1)  

model_base = Net().to(DEVICE)  
optimizer = optim.Adam(model_base.parameters(), lr=0.001) 
```

<br>

### 2-5. 모델 학습을 위한 함수
```python
def train(model, train_loader, optimizer):
    model.train()  
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE) 
        optimizer.zero_grad() 
        output = model(data)  
        loss = F.cross_entropy(output, target) 
        loss.backward()  
        optimizer.step()  
```

<br>

### 2-6. 모델 평가를 위한 함수
```python
def evaluate(model, test_loader):
    model.eval()  
    test_loss = 0 
    correct = 0   
    
    with torch.no_grad(): 
        for data, target in test_loader:  
            data, target = data.to(DEVICE), target.to(DEVICE)  
            output = model(data) 
            
            test_loss += F.cross_entropy(output,target, reduction='sum').item() 
 
            
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item() 
   
    test_loss /= len(test_loader.dataset) 
    test_accuracy = 100. * correct / len(test_loader.dataset) 
    return test_loss, test_accuracy  
```

<br>

### 2-7. 모델 학습
CPU를 사용하여 학습을 진행했기 때문에, 총 학습시간은 2시간 반정도가 걸렸습니다.
```python
import time
import copy
 
def train_baseline(model ,train_loader, val_loader, optimizer, num_epochs = 30):
    best_acc = 0.0  
    best_model_wts = copy.deepcopy(model.state_dict()) 
 
    for epoch in range(1, num_epochs + 1):
        since = time.time()  
        train(model, train_loader, optimizer)
        train_loss, train_acc = evaluate(model, train_loader) 
        val_loss, val_acc = evaluate(model, val_loader)
        
        if val_acc > best_acc: 
            best_acc = val_acc 
            best_model_wts = copy.deepcopy(model.state_dict())
        
        time_elapsed = time.time() - since 
        print('-------------- epoch {} ----------------'.format(epoch))
        print('train Loss: {:.4f}, Accuracy: {:.2f}%'.format(train_loss, train_acc))   
        print('val Loss: {:.4f}, Accuracy: {:.2f}%'.format(val_loss, val_acc))
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
    model.load_state_dict(best_model_wts)  
    return model
 

base = train_baseline(model_base, train_loader, val_loader, optimizer, EPOCH)  	 #(16)
torch.save(base,'baseline.pt')
```
```
-------------- epoch 1 ----------------
train Loss: 1.6946, Accuracy: 53.35%
val Loss: 1.7151, Accuracy: 52.90%
Completed in 5m 59s
-------------- epoch 2 ----------------
train Loss: 1.0632, Accuracy: 68.19%
val Loss: 1.1031, Accuracy: 67.37%
Completed in 5m 18s

...
(중략)
...

-------------- epoch 29 ----------------
train Loss: 0.0645, Accuracy: 98.43%
val Loss: 0.2199, Accuracy: 93.08%
Completed in 5m 57s
-------------- epoch 30 ----------------
train Loss: 0.0845, Accuracy: 97.58%
val Loss: 0.2538, Accuracy: 91.95%
Completed in 5m 53s
```

## 정리
### 1. Net(nn.Module)
CNN의 구조를 정의 해주었다. 
정의한 CNN 모델 Net()의 새로운 객체를 생성하여 현재 사용중인 장비에 할당하였다. 
해당 모델은 model_base에 저장하였다. 
또한, optimizer는 Adam으로 설정하고, 학습률은 0.001로 설정하였다.
```python
model_base = Net().to(DEVICE)  
optimizer = optim.Adam(model_base.parameters(), lr=0.001) 
```
<br>

### 2. train(model, batch, optimizer)
모델 학습을 위한 일련의 과정을 담고 있다.
model을 학습 모드로 설정하며, data와 target을 사용중인 장비에 할당하여 손실함수를 구한다. 
분류 문제이므로 손실함수는 Cross Entropy Loss를 사용한다. 
손실함수를 바탕으로 Gradient를 구하여, 손실함수를 최소로 하도록 Parameter를 업데이트 해주는 함수이다. 

<br>

### 3. evaluate(model, batch)
모델 평가를 위한 일련의 과정을 담고있다. 
model를 평가 모드로 설정하며, data와 target을 사용중인 장비에 해당하여 총 loss 값과 총 정답 수를 구한다. 
해당 값을 전체 batch 수로 나누어 loss와 정확도의 평균을 구하여, 이를 반환해주는 함수이다.

<br>

### 4. Train(model, batch1, batch2, optimizer, num_epoch)
정확도가 가장 높은 모델과 그 수치를 저장하는 변수 best_model_wts, best_acc를 각각 초기화 한다. 
이후, Train Batch로 모델을 학습하고, Train Batch와 Val Batch의 정확도와 loss를 구한다. 
best_acc보다 val batch의 정확도가 더 높으면 best_acc와 best_model_wts를 업데이트한다. 
해당 과정을 epoch 수만큼 반복하여 최적의 모델을 학습하고, 학습이 완료된 모델을 저장한다.

```python
base = train_baseline(model_base, train_loader, val_loader, optimizer, EPOCH)  	 #(16)
torch.save(base,'baseline.pt')
```
