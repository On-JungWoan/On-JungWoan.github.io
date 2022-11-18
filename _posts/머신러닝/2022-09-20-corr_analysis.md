---
title:  "[ML]치과 데이터 상관분석"
excerpt: "논문에 활용할 상관분석 자료 만들기"

categories:
  - ML
tags:
  - [ML, 상관분석]

published: true

toc: true
toc_sticky: true
 
date: 2022-09-20
last_modified_at: 2022-09-20
---

주변 지인이 논문에 사용할 상관분석을 부탁해서 해드렸다.
분석하고자 하는 내용은 다음과 같다.

* **유착치의 위치에 따른 유병률**
* **유착치와 치아 결손간의 상관관계**

<br>
<br>

## 1. Import Libarary & Load DataSet

작업환경은 colab이고 필요한 파일은 구글 드라이브에 업로드하여 사용하였다.
한글 인코딩 값은 cp949를 사용하였으며, 데이터셋은 개인정보 문제로 따로 소개하지 않겠다.

```python
# data handling
import pandas as pd

# p-value
import scipy.stats as stats
```

```python
url ="/content/dent.csv"

df = pd.read_csv(url, encoding="cp949")
df = df.fillna(0)
```

<br>
<br>

## 2. Data Pre-Processing

### 2-1) 치아 결손 관련 전처리

원시 데이터에는 치아의 위치만 표기되어 있어서 위치를 개수로 바꾸어주는 작업을 하였다.

```python
for i in range(4,9):
    df[df.columns[i]] = df.iloc[:,i].apply(lambda x: 0 if x==0 else x.count('/')+1)
```

<br>

### 2-2) 유병률 관련 전처리

병의 유무만 판단하기 위해서 병의 개수가 1 이상이면 전부 1로 취급하였다.

```python
df['그 외 치아 형태 이상 or 수이상'] = df['그 외 치아 형태 이상 or 수이상'].apply(lambda x: 1 if x!=0 else 0)
df['동반하는 합병증'] = df['동반하는 합병증'].apply(lambda x: 1 if x!=0 else 0)    
```

<br>

### 2-3) 합계 변수 생성

결손과 유착치의 총 개수에 대한 파생변수를 만들어주었다.

```python
df['유착치_total'] = df[['상악 치아 전체', '하악 제1유구치','하악제2유구치']].sum(axis=1)
df['결손_total'] = df[['계승치 결손', '다른 영구치 결손', '그 외 치아 형태 이상 or 수이상']].sum(axis=1)
```

<br>
<br>

## 3. 분석

### 3-1) 유착치의 위치에 따른 유병률

* **상관분석**

```python
disease_corr = df[['상악 치아 전체', '하악 제1유구치', '하악제2유구치', '유착치_total', '동반하는 합병증']].corr()
disease_corr = disease_corr.loc[['동반하는 합병증'], ['상악 치아 전체', '하악 제1유구치', '하악제2유구치', '유착치_total']]

disease_vis = pd.DataFrame(disease_corr.unstack()[disease_corr.unstack() != 1].sort_values(ascending=False), columns=['corr'])
disease_vis.style.background_gradient(cmap='viridis')
```

![image](https://user-images.githubusercontent.com/84084372/191171310-e3a49684-4ebc-4267-9a0f-1e62acfbdbe0.png)

<br>

* **유병률**

```python
def makeProb(col):
  tmp = disease_df[col >= 1]
  return tmp['동반하는 합병증'].sum()/len(tmp)

disease_df = df[['상악 치아 전체', '하악 제1유구치', '하악제2유구치', '유착치_total', '동반하는 합병증']]

for col in disease_df:
  if col=='동반하는 합병증':
    continue

  print("{} : {}%".format(col, round(makeProb(df[col])*100, 2)))
```

```python
상악 치아 전체 : 33.33%
하악 제1유구치 : 18.18%
하악제2유구치 : 44.44%
유착치_total : 29.79%
```

<br>

* **해석**
  * 상악 치아에 유착치가 있을 경우,  동반하는 합병증이 발생할 확률은 약 33%입니다.
  * 하악 제 1유구치에 유착치가 있을 경우,  동반하는 합병증이 발생할 확률은 약 18%입니다.
  * 하악 제 2유구치에  유착치가 있을 경우,  동반하는 합병증이 발생할 확률은 약 44%입니다.
  * 유착치가 하나라도 존재하는 모든 case에서,  동반하는 합병증이 발생할 확률은 약 30%입니다.

<br>

### 3-2) 치아 결손과의 상관관계 분석

* **상관계수 계산**

```python
break_corr = df[['상악 치아 전체', '하악 제1유구치', '하악제2유구치', '유착치_total', '계승치 결손', '다른 영구치 결손', '결손_total']].corr()
break_corr = break_corr.loc[['계승치 결손', '다른 영구치 결손', '결손_total'], ['상악 치아 전체', '하악 제1유구치', '하악제2유구치', '유착치_total']]

break_vis = pd.DataFrame(break_corr.unstack()[break_corr.unstack() != 1].sort_values(ascending=False), columns=['corr'])
```

<br>

* **p-value 계산**

```python
p_list = []

for row in break_corr.index.tolist():
  tmp = []
  for col in break_corr.columns.tolist():
    tmp.append(stats.pearsonr(df.loc[:,row], df.loc[:,col])[1])
  p_list.append(tmp)

tmp_df = pd.DataFrame(p_list, index=['계승치 결손', '다른 영구치 결손', '결손_total'], columns=['상악 치아 전체', '하악 제1유구치', '하악제2유구치', '유착치_total'])

tmp_df = pd.DataFrame(tmp_df.unstack(), columns=['p_value'])
tmp_df

# join 및 출력
break_vis = break_vis.join(tmp_df)
break_vis
```

![image](https://user-images.githubusercontent.com/84084372/191171803-aef1744c-0b20-4060-8d07-10fa50e2166e.png)

<br>

* **해석**
  * 하악 제1,2 유구치에는 치아 결손과의 상관관계가 존재하지 않습니다.
  * 상악 치아의 유구치 개수는 다른 영구치의 결손 수, 총 결손 수와 양의 상관관계가 존재하며, 계승치의 결손 수와는 약한 양의 상관관계가 존재합니다.
  * 이 때의 p-value는 최대 약 0.008로 유의수준 1% 내에서 유의한 결과입니다.

<br>
<br>

## 4. 분석 방법에 대한 기술

### 4-1) 용어 정의

해당 분석에서 사용한 용어에 대한 정의이다.

```
- 유착치 : 상악 치아 전체, 하악 제1 유구치,  하악 제2 유구치
- 치아 결손 : 계승치 결손, 다른 영구치 결손
- 병의 유무 : 동반하는 합병증의 유무
- 결손_total : ‘계승치 결손’ + ‘다른 영구치 결손’ + ‘그 외 형태이상 or  수 이상’
- 유착치_total : ‘상악 치아 전체’ + ‘하악 제1 유구치’ + ‘하악 제2 유구치’
```

<br>

### 4-2) 데이터 전처리

**상악 치아 전체** ,  **하악 제 1 유구치**,  \*\*하악 제 2 유구치 \*\* , \*\*계승치 결손 \*\*,  **다른 영구치 결손**에 해당하는 관측치는 count 값으로 변환하여 분석을 진행하였다. 유병률에서 **병**은 통반하는 합병증의 유무로 변환하였으며, 1과 0의 값을 가짐.

<br>

### 4-3) 유병률

유병률은 특정 조건을 만족하는 표본에서 통반하는 합병증이 존재하는 case의 비율을 계산

<br>

### 4-4) 상관분석

상관분석은 pearson 상관계수를 사용하였으며, 사용한 기준은 다음과 같다.

| 0.0 \~ 0.2 | 상관관계 거의 없음 |
| --- | --- |
| 0.2 \~ 0.4 | 낮은 상관관계 |
| 0.4 \~ 0.6 | 상관관계가 있음 |
| 0.6 \~ 0.8 | 강한 상관관계 |
| 0.8 \~ 1.0 | 매우 강한 상관관계 |

<br>



{% capture date %}{{page.date | remove: '-'}}{% endcapture %}
{% capture month %}{{ date | slice: 4, 2 }}{% endcapture %}
{% capture month_day %}{{ date | slice: 4, 4 }}{% endcapture %}

<div align='center'>
  <a href="https://on-jungwoan.github.io/til/TIL-{{month}}/#{{month_day}}">
    본문으로 돌아가기
  </a>
</div>