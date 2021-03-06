---
title:  "[딥러닝] 작물 잎 사진으로 질병 분류하기 예제 풀이 #2"
excerpt: "교재 : 파이토치 딥러닝 프로젝트 문제집"

categories:
  - DL
tags:
  - [딥러닝, CNN, 전이학습]

published: true

toc: true
toc_sticky: true
 
date: 2022-07-28
last_modified_at: 2022-07-30
---

<br>

학습 데이터 셋의 크기가 큰 관계로, CPU를 사용하여 학습 시 학습 시간이 너무 오래걸렸다. 
따라서 colab에서 작업을 해주어야 했는데, 데이터셋의 크기가 너무 커서 드라이브에 올리는 데 7시간이나 걸린다는 것이다. 
따라서 colab에서 원본 데이터셋 zip파일의 링크를 걸어 다운로드 받은 뒤, colab 환경 내에서 압축 해제를 진행하였다. 
드라이브에서 작업을 할때는 속도 제한이 걸리지만, colab에서 작업할 떄는 속도 제한이 걸리지 않아 3분 남짓의 시간 안에 성공적으로 드라이브에 업로드 하였다. 
colab에서 사용하기 위해 코드를 조금 바꾸어주었다. 

```python
# colab에서 사용시

from google.colab import drive
drive.mount('/content/drive')
```
```python
# 데이터 다운로드
!gdown https://drive.google.com/uc?id=1uBY-JbXcPd-tikzFJcbwSR9_zEjYHHor
!mv /content/dataset.zip /content/drive/MyDrive/04_작물_잎_질병/dataset

# 압축 풀기
%cd /content/drive/MyDrive/04_작물_잎_질병/dataset
!unzip -qq "/content/drive/MyDrive/04_작물_잎_질병/dataset/dataset.zip"
```
```python
# laptop에서 작업 시
# working_dir = "D:/python_projects/deep-learning-with-projects/datasets/04_작물_잎_사진_질병_분류/"

# jnu pc에서 작업 시
# working_dir = "C:/Users/user/Downloads/"

# colab에서 작업 시
working_dir = "/content/drive/MyDrive/04_작물_잎_질병/"
```

<br>

## 전이학습이란?
ImageNet 훈련 데이터를 이용하여 Pre-Trained된 Model을 사용한다. 
해당 파라미터 값 일부를 우리가 가진 데이터셋의 특성에 맞게 조정하며, 이를 Fine-Tuning 이라고 한다. 
여기서 유의해야 할 점은, 모든 파라미터를 업데이트 하진 않는다는 것이다. 
다른 종류의 이미지라도 낮은 수준의 특징은 상대적으로 비슷할 가능성이 높기 때문이다. 
따라서 분류기와 가까운 Layer부터 원하는 만큼 학습 과정에서 업데이트하며, 나머지 Layer는 Freeze한다. 
이 때, Freeze Layer 수는 pre-trained model에 사용된 데이터셋과의 유사성을 고려하여 결정한다. 
다음 사진은, Freeze Layer의 수를 결정하는 전략이다. 

![image](https://user-images.githubusercontent.com/84084372/181512540-12cc7327-f854-4ba3-8a54-c7990caed605.png)

- 데이터 수가 많고, 이용된 데이터와의 유사도 높음 (Quadrant2)
    - 분류기와 가까운 일부 Layer만을 훈련시키고, 나머지는 Freeze 한다.
    - 데이터의 유사도가 높기 때문에 상대적으로 적은 수의 Layer만 업데이트 해도 성능이 좋음

- 데이터 수가 적고, 이용된 데이터와의 유사도 높음 (Quadrant4)
    - Convolutional Base 부분 전체를 Freeze하고, 분류기 부분만 변경한다.
    - 일부 Layer를 재학습하기에는 오버피팅의 위험 높음.

- 데이터 수가 많고, 이용된 데이터와의 유사도 낮음 (Quadrant1)
    - 전체 데이터를 Unfreeze하여 학습을 진행
    - 모든 Layer가 학습 과정에서 업데이트 되지만, pre-trained 모델의 효과적인 구조는 그대로 유지 됨.

- 데이터 수가 적고, 이용된 데이터와의 유사도 적음 (Quadrant3)
    - 일부 Layer를 freeze하고, 나머지는 전부 학습한다.
    - 모든 Layer를 전부 Unfreeze하기에는, 데이터 크기가 작기 때문에 오버피팅의 위험이 있다.

<br>

## 3. Pre-Trained Model Fine Tuning
### 3-1. Transfer Learning을 위한 준비
```python
# colab에서 사용시

from google.colab import drive
drive.mount('/content/drive')

working_dir = "/content/drive/MyDrive/04_작물_잎_질병/"
```
```python
import torch
import os
 
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# 하이퍼 파라미터 정의
BATCH_SIZE = 256 
EPOCH = 30 
```
```python
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder 

# transforms 정의(train과 val을 다르게 정의)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([64,64]), # 이미지 사이즈를 64*64로 조정

        # 이미지 Augmentation
        transforms.RandomHorizontalFlip(), # 이미지 좌우반전 (default) p = 0.5
        transforms.RandomVerticalFlip(),  # 이미지 상하반전 (default) p = 0.5
        transforms.RandomCrop(52), # 이미지의 랜덤한 부위를 잘라내어 52*52로 만듦
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) # 정규화를 진행할 ImageNet의 평균과 표편
        ]),
    
    'val': transforms.Compose([
        transforms.Resize([64,64]),  
        transforms.RandomCrop(52), transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]) 
        ])
}
```
```python
data_dir = working_dir + 'splitted'

# ImageFolder를 사용하여 이미지 불러오기
image_datasets = {x: ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']} 

# 미니 Batch 구성
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in ['train', 'val']} 

# 데이터셋 사이즈
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# 클래스명
class_names = image_datasets['train'].classes
```

<br>

### 3-2. Pre-Trained Model 불러오기
```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
 
resnet = models.resnet50(pretrained=True)  # False로 설정시 모델의 구조만 가져오고 파라미터 값은 랜덤으로 설정
print(resnet.fc) # 기존 resnet은 in 2048에 out 1000임

num_ftrs = resnet.fc.in_features   
resnet.fc = nn.Linear(num_ftrs, 33) 
resnet = resnet.to(DEVICE)
 
criterion = nn.CrossEntropyLoss() 
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=0.001)
 
from torch.optim import lr_scheduler
# 7 epoch마다 학습률에 0.1씩을 곱해줌
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) 
```

<br>

### 3-3. Pre-Trained Model의 일부 Layer Freeze하기
resnet은 총 10개의 Layer로 이루어져 있음.
```python
cnt = 0
for c in resnet.children():
  cnt += 1

print(cnt)
```
```
10
```
<br>

```python
ct = 0 
for child in resnet.children():  
    ct += 1  
    if ct < 6: 
        for param in child.parameters():
            param.requires_grad = False
```

<br>

### 3-4. Transfer Learning 모델 학습과 검증을 위한 함수
```python
import time
import copy

def train_resnet(model, criterion, optimizer, scheduler, num_epochs=25):

    best_model_wts = copy.deepcopy(model.state_dict())  
    best_acc = 0.0  
    
    for epoch in range(num_epochs):
        print('-------------- epoch {} ----------------'.format(epoch+1)) 
        since = time.time()                                     
        for phase in ['train', 'val']: 
            if phase == 'train': 
                model.train() 
            else:
                model.eval()     
 
            running_loss = 0.0  
            running_corrects = 0  
 
            
            for inputs, labels in dataloaders[phase]: 
                inputs = inputs.to(DEVICE)  
                labels = labels.to(DEVICE)  
                
                optimizer.zero_grad() 
                
                with torch.set_grad_enabled(phase == 'train'):  
                    outputs = model(inputs)  
                    _, preds = torch.max(outputs, 1) 
                    loss = criterion(outputs, labels)  
    
                    if phase == 'train':   
                        loss.backward()
                        optimizer.step()
 
                running_loss += loss.item() * inputs.size(0)  
                running_corrects += torch.sum(preds == labels.data)  
            if phase == 'train':  
                scheduler.step()
 
            epoch_loss = running_loss/dataset_sizes[phase]  
            epoch_acc = running_corrects.double()/dataset_sizes[phase]  
 
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) 
 
          
            if phase == 'val' and epoch_acc > best_acc: 
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
 
        time_elapsed = time.time() - since  
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
 
    model.load_state_dict(best_model_wts) 

    return model
```

<br>

### 3-5. 모델 학습을 실행하기
```python
model_resnet50 = train_resnet(resnet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=EPOCH) 

torch.save(model_resnet50, 'resnet50.pt')
```
```
-------------- epoch 1 ----------------
train Loss: 0.5778 Acc: 0.8258
val Loss: 0.3665 Acc: 0.8908
Completed in 27m 41s <- 노트북 코어수를 고려하지 않고 worker 수를 설정해서 학습 시간 매우 길어짐(2로 수정)
-------------- epoch 2 ----------------
train Loss: 0.2227 Acc: 0.9279
val Loss: 0.1775 Acc: 0.9440
Completed in 1m 16s
...
(중략)
...
-------------- epoch 29 ----------------
train Loss: 0.0114 Acc: 0.9961
val Loss: 0.0322 Acc: 0.9897
Completed in 1m 13s
-------------- epoch 30 ----------------
train Loss: 0.0115 Acc: 0.9965
val Loss: 0.0292 Acc: 0.9903
Completed in 1m 14s
Best val Acc: 0.990278
```

<br>

## 4. 모델 평가
### 4-1. 베이스라인 모델 평가를 위한 전처리
```python
# transforms 정의
transform_base = transforms.Compose([transforms.Resize([64,64]),transforms.ToTensor()])

# 데이터 불러오기
test_base = ImageFolder(root=working_dir + 'splitted/test',transform=transform_base)  

# 미니배치 구성
test_loader_base = torch.utils.data.DataLoader(test_base, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
```

<br>

### 4-2. Transfer Learning 모델 평가를 위한 전처리
```python
# transforms 정의
transform_resNet = transforms.Compose([
        transforms.Resize([64,64]),  
        transforms.RandomCrop(52),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

# 데이터 불러오기    
test_resNet = ImageFolder(root=working_dir + 'splitted/test', transform=transform_resNet) 

# 미니배치 구성
test_loader_resNet = torch.utils.data.DataLoader(test_resNet, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
```

<br>

### 4-3. 베이스 라인 모델 평가
베이스라인 모델은 cpu 환경에서 학습했기 때문에, device로 cuda를 사용하면 type 에러가 발생한다. 
따라서 학습했을 당시의 device로 맞추어주어야 한다.
```python
DEVICE = torch.device("cpu")
```
```python
baseline = torch.load(working_dir + 'model/baseline.pt') 
baseline.eval()  
test_loss, test_accuracy = evaluate(baseline, test_loader_base)

print('baseline test acc:  ', test_accuracy)
```
```
baseline test acc:   96.52237354085604
```

<br>

### 4-4. Transfer Learning 모델 평가
```python
DEVICE = torch.device("cuda")
```
```python
resnet50=torch.load(working_dir + 'model/resnet50.pt') 
resnet50.eval()  
test_loss, test_accuracy = evaluate(resnet50, test_loader_resNet)

print('ResNet test acc:  ', test_accuracy)
```
```
ResNet test acc:   98.88132295719845
```
