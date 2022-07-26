---
title:  "[딥러닝] MNIST 예제 풀이"
excerpt: "교재 : 파이토치 딥러닝 프로젝트 문제집"

categories:
  - DL
tags:
  - [딥러닝, CNN]

published: true

toc: true
toc_sticky: true
 
date: 2022-07-26
last_modified_at: 2022-07-26
---

<br>

사용교재 : 한 줄씩 따라 해보는 파이토치 딥러닝 프로젝트 모음집 <br>
예제 : MNIST를 사용한 손글씨 숫자 이미지 분류 문제

<br>

## 0. Library
이 예제는 PyTorch를 사용하여 학습을 진행한다.<br>
Colab을 사용했기 때문에 drive 마운트 과정이 포함되어 있다.

```python
# model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# visualizing
import matplotlib.pyplot as plt

# etc
from google.colab import drive
drive.mount('/content/drive')
```

<br>

## 1. 분석 환경 설정
Colab에서 제공하는 GPU를 사용한다.<br>
GPU의 병렬 처리를 위해 device에 cuda를 지정해준다.

```python
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

print('Cuda Device :', device)
```

<br>

## 2. HyperParameter 지정

```python
batch_size = 50
epoch_num = 15
learning_rate = 0.0001
```

<br>

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/180927694-a0ade14d-20b8-4039-a4fb-8d116dce976f.png"></p>
<p align="center">[출처] https://www.slideshare.net/w0ong/ss-82372826</p>

<br>

## 3. 데이터 불러오기
### 3.1 데이터 설명
MNIST 데이터는 이미지와 레이블로 구성되어 있다. <br>
이미지는 [1, 28, 28]의 3차원 행렬이며 각각 [Channel, Width, Height]를 나타낸다(Channel이 1이면 흑백, 3이면 RGB 채널). <br>
레이블은 One-Hot Encoding 방식으로 길이가 10인 벡터로 이루어져 있음. <br>
이미지 데이터는 0에서 1까지의 값을 갖는 고정 크기의 28x28행렬이다. <br>
각 행렬의 원소는 픽셀의 밝기 정보를 나타냄. <br>
1에 가까울수록 흰색, 0에 가까울수록 검은색 픽셀이다. <br>

<br>

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/180925688-7a6c24e4-83a0-4444-a4f8-6bde5e777afa.png"></p>
<p align="center">[출처] http://ai4school.org/?page_id=4389</p>

<br>

### 3.2 데이터 불러오기
해당 데이터가 로컬 폴더 안에 있지 않은 경우, download 옵션을 True로 지정하고, 해당 파일의 경로를 지정해주어야 한다. 
또한, Pytorch는 입력 데이터로 Tensor를 사용하므로 이미지를 Tensor로 변환하는 전처리인 torchvision의 transforms.ToTenser()를 사용한다. 
Train 데이터는 6만개, Test 데이터는 1만개의 관측값으로 이루어져 있다.

```python
train_data = datasets.MNIST(root = '/content/drive/MyDrive/MNIST', train = True, download = False, transform = transforms.ToTensor())
test_data = datasets.MNIST(root = '/content/drive/MyDrive/MNIST', train = False, transform = transforms.ToTensor())

print(len(train_data), len(test_data))
```
```
60000 10000
```

<br>

MNIST 데이터는 앞서 말했듯이, 이미지와 라벨 데이터로 이루어져 있고, 
이미지 데이터는 단일 채널의 [1, 28, 28] 3차원 텐서이다. 
이를 Matplotlib을 사용하여 그려주기 위해, squeeze() 함수를 사용한다. 
이 때, squeeze() 함수는 크기가 1인 차원을 없애는 함수로, 이미지를 [28,28]의 2차원 Tensor로 만들어준다.

```python
image, label = train_data[0]

plt.imshow(image.squeeze().numpy(), cmap = 'gray')
plt.title('label : %s' % label)
plt.show()
```
<p align="center"><img src="https://user-images.githubusercontent.com/84084372/180844603-12136ad5-a8f2-42ff-a828-9b7264389dfe.png"></p>

<br>

## 4. 미니 배치 구성
사이즈가 50인 미니배치를 생성한다. 
Train 데이터셋 크기가 6만개이므로 6만/50 = 1200개의 train 미니배치가 생성된다. 
이 때, 시계열 데이터가 아닌 경우 딥러닝이 데이터의 순서에 대해서는 학습하지 못하도록 필수적으로 shuffle 해준다.

```python
train_loader = torch.utils.data.DataLoader(dataset = train_data, 
                                           batch_size = batch_size, shuffle = True)
test_loader  = torch.utils.data.DataLoader(dataset = test_data, 
                                           batch_size = batch_size, shuffle = True)

first_batch = train_loader.__iter__().__next__()
print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))
print('{:15s} | {:<25s} | {}'.format('Num of Batch', '', len(train_loader)))
print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))
print('{:15s} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape))
print('{:15s} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape))
```

```
name            | type                      | size
Num of Batch    |                           | 1200
first_batch     | <class 'list'>            | 2
first_batch[0]  | <class 'torch.Tensor'>    | torch.Size([50, 1, 28, 28])
first_batch[1]  | <class 'torch.Tensor'>    | torch.Size([50])
```

<br>

각 미니 배치는 [1,28,28]의 이미지가 50개 쌓여있는 형태와, 레이블로 구성되어 있다.
각 배치의 0번 원소는 [50,1,28,28] 형태의 4차원 Tensor이며, Train 배치에는 이 것이 1200개 존재한다.
또한, 각 배치의 1번 원소는 이에 해당하는 레이블이 50개 존재한다.

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/180929620-2f0e3216-31f1-412e-bf37-cdbe0cc36557.png"></p>

```python
print('{:23s} |    {}'.format('first_batch[0][0]', first_batch[0][0].shape))
print('{:23s} |    {}'.format('first_batch[0][0][0]', first_batch[0][0][0].shape))
```
```
first_batch[0][0]       |    torch.Size([1, 28, 28])
first_batch[0][0][0]    |    torch.Size([28, 28])
```

<br>

```python
plt.imshow(first_batch[0][0][0])
plt.title('label : %s' % first_batch[1][0])
plt.show()
```

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/180845046-dc5769e5-2ff5-400b-8171-7078c5d8655c.png"></p>

<br>

## 5. CNN 구조 설계
### 5.1 CNN 구조 정의
이번 CNN은 다음 그림과 같이 2개의 Convolutional Layer와 2개의 Fully-Connected Layer로 설계. 
데이터의 형태를 나타낼때는 대괄호 '[]', 가중치인 Filter 형태를 나타낼 때는 소괄호'()'를 사용, @는 filter의 개수를 나타냄. 
padding(0)과 stride(1)는 기본값으로 설정. 

- Tensor의 채널
  - filter의 개수를 따름

<br>

- Tensor의 가로와 세로의 크기
$$O = \frac{I + 2P - F}{S} + 1 = \frac{28 + 2 \times 0 - 3}{1} + 1 = 26$$

<br>

- MaxPooling
  - Tensor의 가로 세로에만 영향을 주므로, 채널은 동일하고 가로 세로 길이만 반감

<br>

- Flatten
  - Fully-connected Layer 연산을 위해 고차원 Tensor를 1차원으로 줄임
  - MNIST는 0부터 9까지 10개의 클래스이기 때문에 최종적으로 10개의 길이로 구성
  - (64x12x12, 1) -> (128, 1) -> (10, 1)

<br>

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/180930637-9f0deacc-88b8-46e6-aac8-d3678aa6ec00.png"></p>
<p align="center">[출처] 한 줄씩 따라해보는 파이토처 딥러닝 프로젝트 모음집</p>

### 5.2 CNN 구조 설계
```python
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)         # conv1 통과
    x = F.relu(x)             # ReLU 활성함수 적용
    x = self.conv2(x)         # conv2 통과
    x = F.relu(x)             # ReLU 활성함수 적용

    x = F.max_pool2d(x,2)     # 2*2 filter로 max-pooling
    x = self.dropout1(x)      # 사전에 정의한 0.25 확률의 dropout1을 반영
    x = torch.flatten(x, 1)   # 고차원 torch를 1차원 벡터로 변환 = 64*12*12
    x = self.fc1(x)           # 9216 크기의 벡터를 128 크기의 벡터로 학습하는 fc1 통과
    x = F.relu(x)             # ReLU 활성함수 적용
    x = self.dropout2(x)      # 사전에 정의한 0.5 확률의 dropout1을 반영
    x = self.fc2(x)           # 128 크기의 벡터를 10 크기의 벡터로 학습하는 fc2 통과

    # 최종 출력값 : log-softmax를 사용한다.
    # softmax가 아닌 log_softmax를 사용하면 연산 속도를 높일 수 있다.
    output = F.log_softmax(x, dim=1)

    return output
```

<br>

## 6. Optimizer 및 손실 함수 정의
```python
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss( )
```

<br>

## 7. 모델 학습
```python
model.train()
i = 1
for epoch in range(epoch_num):
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        # 이전 반복 시행에서 저장된 optimizer의 gradient를 초기화
        optimizer.zero_grad()

        # Feed Forward (미니 배치를 모델에 통과시킴)
        output = model(data)
        
        # 손실함수를 통해 Gradient 계산
        loss = criterion(output, target)
        loss.backward()

        # Gradient를 통해 모델의 가중치 업데이트
        optimizer.step()

        # 학습 1000번마다 손실함수 출력
        if i % 1000 == 0:
            print('Train Step: {}\tLoss: {:.3f}'.format(i, loss.item()))
        i += 1
```
```
Train Step: 1000	Loss: 0.213
Train Step: 2000	Loss: 0.252
Train Step: 3000	Loss: 0.057
Train Step: 4000	Loss: 0.216
Train Step: 5000	Loss: 0.008
Train Step: 6000	Loss: 0.085
Train Step: 7000	Loss: 0.018
Train Step: 8000	Loss: 0.020
Train Step: 9000	Loss: 0.014
Train Step: 10000	Loss: 0.072
Train Step: 11000	Loss: 0.007
Train Step: 12000	Loss: 0.019
Train Step: 13000	Loss: 0.091
Train Step: 14000	Loss: 0.069
Train Step: 15000	Loss: 0.172
Train Step: 16000	Loss: 0.071
Train Step: 17000	Loss: 0.088
Train Step: 18000	Loss: 0.016
```        
<br>

## 8. 모델 평가

```python
model.eval()
correct = 0
for data, target in test_loader:
#     data, target = Variable(data, volatile=True), Variable(target)
    data = data.to(device)
    target = target.to(device)
    output = model(data)

    # log-softmax 값이 가장 큰 인덱스를 예측값으로 저장.
    prediction = output.data.max(1)[1]

    
    correct += prediction.eq(target.data).sum()

print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))
```
```
Test set: Accuracy: 98.97%
```
