---
title:  "[ML] 아파트 실거래가 예측 (1)"
excerpt: "아파트 실거래가 예측 Toy Proejct"

categories:
  - toy_project
tags:
  - [ML, ICT인턴십, orange3]

published: true

toc: true
toc_sticky: true
 
date: 2022-10-15
last_modified_at: 2022-10-15
---

## 1. Make Train/Test Data
### 1-1) 데이터 수집

![image](https://user-images.githubusercontent.com/84084372/196024163-916a9bb3-82d9-469e-ba7c-968e7c9f1993.png)

원시 데이터는 국토교통부 실거래가 공개시스템에서 받아주었다. 기간은 2019년 9월 30일부터 2022년 09월 30일까지 3개년 데이터를 사용하였다.

![image](https://user-images.githubusercontent.com/84084372/196024312-28b9f9ef-ef47-48b3-80d8-8e33ab32c4cf.png)

15개의 컬럼과, 약 7만개의 관측치로 이루어져있는 데이터이다.

<br>

### 1-2) Import Library & Load Dataset

전처리 작업을 위해 원시 데이터를 불러와주었다. 한글 인코딩은 cp949를 사용하였다.

```python
# Data Handling
import pandas as pd
import numpy as np
import os

# Model Selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

```python
FILE_PATH = '/content/drive/MyDrive/apt'
FILE_NAME = '2019_2022_매매데이터.csv'
TRAIN_SAVE_PATH = '/content/drive/MyDrive/apt'
TEST_SAVE_PATH = '/content/drive/MyDrive/apt'
POP_FILE_PATH = '/content/drive/MyDrive/apt'
POP_FILE_NAME = 'pop.csv'
```

```python
df = pd.read_csv(os.path.join(FILE_PATH, FILE_NAME), encoding='cp949')
```

![image](https://user-images.githubusercontent.com/84084372/196024529-68a48ca2-0661-42c9-a7a4-0e90dc58985c.png)

<br>

구별, 동별 요약통계량을 추가해주기 위해 구 데이터를 따로 추출해주었다.

```python
df['구'] = df['시군구'].apply(lambda x: x.split(' ')[1])
```

<br>

### 1-3) 요약통계량 추가
구별, 동별 거래금액에 대한 요약통계량을 추가해주었다.

```python
dong_mean_money = pd.DataFrame(df.groupby(['시군구']).mean()['거래금액(만원)'].round().astype('int'))
dong_sum_money = pd.DataFrame(df.groupby(['시군구']).sum()['거래금액(만원)'].round().astype('int'))

dong_mean_money.columns = ['동별_평균_거래금액']
dong_sum_money.columns = ['동별_총_거래금액']

df = pd.merge(left=df, right=dong_mean_money, on='시군구')
df = pd.merge(left=df, right=dong_sum_money, on='시군구')
```

```python
gu_mean_money = pd.DataFrame(df.groupby(['구']).mean()['거래금액(만원)'].round().astype('int'))
gu_sum_money = pd.DataFrame(df.groupby(['구']).sum()['거래금액(만원)'].round().astype('int'))

gu_mean_money.columns = ['구별_평균_거래금액']
gu_sum_money.columns = ['구별_총_거래금액']

df = pd.merge(left=df, right=gu_mean_money, on='구')
df = pd.merge(left=df, right=gu_sum_money, on='구')
```

<br>

### 1-4) 계약년도, 계약월 분리

계약년도와 계약월의 특성이 다르다고 판단하여 분리해주었다. 계약월은 Seasonal Data로, 같은 월끼리 비슷한 양상을 보일 것이라고 판단하였다. 또한, 계약년도가 최근에 가까울수록 거래가도 비례하여 증가한다고 판단하였기 때문에 두 변수간의 서로 다른 특성을 고려하여 분리시켜주었다.

```python
df['계약년'] = df['계약년월'].apply(lambda x: str(x)[:4])
df['계약월'] = df['계약년월'].apply(lambda x: str(x)[4:])
```

<br>

### 1-5) Null값 처리

거래유형은 직거래와 중개거래 2가지 value를 갖는 범주형 변수이다. null값은 최빈값으로 대치해주었다.

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/196024881-4e6db0c9-941e-410a-9282-bf2fed62b4d3.png"></p>

```python
df['거래유형'][ df['거래유형'] == '-' ] = '중개거래'
df['거래유형'].value_counts()
```

<br>

### 1-6) 구별 인구데이터 추가

사실 동별 인구 데이터를 추가하고 싶었으나, 마땅한 법정동과 행정동 매칭 데이터가 없어서 구별 인구 데이터를 추가하였다. 별 영향력은 없을 것으로 예상되지만, 그래도 한번 넣어보았다.

```python
pop = pd.read_csv(os.path.join(POP_FILE_PATH, POP_FILE_NAME), encoding='cp949')
```

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/196025525-3bd8fc01-7325-4b3b-8fbb-f267ea29fdaf.png"></p>

```python
pop['구별'] = pop['구별'].apply(lambda x: x.replace(' ', ''))
pop['인구'] = pop['인구'].apply(lambda x: x.replace(',', '')).astype("int")

df = pd.merge(df, pop, left_on='구', right_on='구별')
```

<br>

### 1-7) Scaling

numeric 데이터에 대해서 스케일링을 해주었다. Feature가 많지는 않아서 효과는 미비할 것으로 예상되지만, 변수들 스케일이 많이 달라서 일단 해주었다. 스케일러는 Standard Scaler를 사용하였다.

```python
scaler = StandardScaler()

num_column = [
    '전용면적(㎡)', '동별_평균_거래금액',
       '동별_총_거래금액', '구별_평균_거래금액', '구별_총_거래금액', '인구'
]

res = scaler.fit_transform(df[num_column])
df[num_column] = pd.DataFrame(res, columns=num_column)
```

<br>

### 1-8) 최종 Data Frame

```python
new_df = df[[
    '전용면적(㎡)', '계약년', '계약월',
       '층', '건축년도', '거래유형', '동별_평균_거래금액',
       '동별_총_거래금액', '구별_평균_거래금액', '구별_총_거래금액', '인구', '거래금액(만원)'
]]

new_df.head(1)
```

![image](https://user-images.githubusercontent.com/84084372/196025721-795b604f-346b-48b0-9d8c-5a4f938529da.png)

<br>

### 1-9) Train / Test Split

test 데이터는 30% 뽑아주었고, 무작위 추출을 해주었다.

```python
train, test = train_test_split(new_df, test_size = 0.3, shuffle=True)

train.to_csv(os.path.join(TRAIN_SAVE_PATH, 'train.csv'), encoding='cp949')
test.to_csv(os.path.join(TEST_SAVE_PATH, 'test.csv'), encoding='cp949')
```

<br>

```python
print(train.shape, test.shape, sep='\n\n')
```
```
(49566, 12)
(21243, 12)
```

<br>
<br>

## 2. Model Evaluate by Orange3

### 2-1) Use Orange3

모델 평가는 Orange3라는 머신러닝 tool을 사용하여 진행해주었다. 대략적으로 모델의 성능을 평가할 때 사용하기 좋은 tool인 것 같다. Orange3에서 지원하는 모델 중 회귀에 대한 Random Forest, Gradient Boosting, Linear Regerssion, AdaBoost 알고리즘으로 성능을 평가해보았다.

<br>

### 2-2) Settings

우선, Train과 Test 데이터를 Orange3에서 불러와주고, Feature 설정을 해주었다. index와 같은 id 데이터는 meta데이터로 설정해주고, 범주 및 numeric 데이터와 target 데이터를 설정해주면 된다.

<img src="https://user-images.githubusercontent.com/84084372/196026113-79f04b49-624f-4468-ab5a-3f197b8e869e.png" width="400px" height="400px">
<img src="https://user-images.githubusercontent.com/84084372/196026117-81c999fc-1856-4773-8763-e7527143d504.png" width="400px" height="400px">

![image](https://user-images.githubusercontent.com/84084372/196026179-a186aaf9-173d-4da4-96b4-ea75f30d1e93.png)

<br>

### 2-3) Evaluate

- **Score 비교**

Score는 AdaBoost가 가장 좋았다. 평균 거래가가 3억 중반정도인데 MAE가 2000만원정도 나왔다. 결정계수도 94%로 준수한 성능을 보여주고 있었다.

![image](https://user-images.githubusercontent.com/84084372/196026198-106e9a49-d5c1-48c1-a741-5f0b61241620.png)

<br>

- **변수 중요도**

변수 중요도도 확인할 수 있다. RReliefF 방식을 사용한다.

![image](https://user-images.githubusercontent.com/84084372/196026239-6cf18f46-c359-499c-b1fe-af181ee3f134.png)

<br>

### 2-4) Predict

광주광역시 북구 일곡동에 위치한 40평 아파트를 2022년 10월에 계약한다고 했을 때의 아파트 거래가를 예측해보았다.

![image](https://user-images.githubusercontent.com/84084372/196027728-02a7951c-1209-4230-9f45-c3b1fce884b8.png)

![image](https://user-images.githubusercontent.com/84084372/196027835-9d431e24-d10d-4666-992e-ff73341767b4.png)

<br>

모델은 예상 거래가로 4억원을 리턴했고, 네이버 부동산에서 조회한 실거래가도 약 4억원대에 형성되어 있었다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/84084372/196028231-5c992441-5858-4160-bd87-300c4dce304e.png" style="border: 2px solid black">
    <strong>  ->  </strong>
    <img src="https://user-images.githubusercontent.com/84084372/196029985-ddd59529-18a2-4ed4-a000-b53d94cefd72.png" style="border: 2px solid black">
</p>

<br>

### 2-5) Model Export

위에서 학습한 model을 pickle 파일 형태로 Export 할 수도 있다.

![image](https://user-images.githubusercontent.com/84084372/196028258-67d3b27d-2875-406b-82ce-7d1dd8964258.png)

다음 코드로 Python에서 사용가능하다.

```python
import joblib
import Orange

MODEL_PATH = 'c:/users/user/desktop/model.pkcls'

model = joblib.load(MODEL_PATH)
```

<br>
<br>

<div align='center'>
  <a href="https://on-jungwoan.github.io/internship/Itern-TIL-10/#1015">
    본문으로 돌아가기
  </a>
</div>    
