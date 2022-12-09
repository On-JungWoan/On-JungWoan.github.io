---
title:  "[Google Indexing API, Github Actions] 구글 서치 콘솔 색인 생성 자동화 하는 법"
excerpt: "How to Automate the Indexing Process on Google Search Console"

categories:
  - Blog
tags:
  - [Githubio, jekyll, html]


published: true

toc: true
toc_sticky: true
 
date: 2022-12-09
last_modified_at: 2022-12-09
til: 'true'
---

## 1. 기존의 페이지 색인 생성 방식 문제점

![image](https://user-images.githubusercontent.com/84084372/206609633-b80d2c83-ff3a-468a-ba36-a86530f6023c.png)

기존에는 페이지 색인을 생성하기 위해, Google Search Console에 들어가서 일일이 URL을 입력하고, 색인 생성 요청을 누른 뒤, 1~2분을 기다려야 했다. 매번 포스팅을 작성할 때마다 이런 번거로운 일을 하기도 귀찮고, 한동안 까먹고 밀리기라도 하면, 매우 끔찍하다... 블로그 조회수와도 직결되는 중요한 작업이기 때문에 안할수도 없어서, 이 귀찮은 과정을 자동화하는 방법에 대해 소개한다.

<br>
<br>

## 2. Google Indexing API

> Indexing API를 사용하면 페이지가 추가되거나 삭제된 경우 사이트 소유자가 Google에 직접 알릴 수 있습니다. 이렇게 하면 Google에서 페이지를 새로 크롤링하도록 예약하여 사용자 트래픽의 질이 향상될 가능성이 있습니다. <br><br> - Google Indexing API

Google Indexing API 소개글의 일부이다. 위 소개글에서도 알 수 있듯이 Google Indexing API는 페이지가 새로 추가되거나 삭제된 경우, Search Console을 사용하지 않고도 간편하게 색인 생성을 요청할 수 있다.

### 2-1) 초기 세팅

#### 2-1-1) 서비스 계정 생성 후 사이트 소유자로 추가

위 API를 사용하기 위해서는 몇 가지의 초기 세팅이 필요하다. 초기 세팅 방법은 아래 가이드북에 자세히 나와있다. 우선, 아래 링크를 클릭하여, `2. 서비스 계정에 소유자 상태 부여` 과정까지 따라하도록 하자. 어렵지 않으니, 차근차근 따라하면 된다. 중간에 json 파일을 열어서 client_email을 찾아야하는데, 마땅한 뷰어가 없을 경우 메모장으로 열면 된다.

- **Link**

  <https://developers.google.com/search/apis/indexing-api/v3/prereqs>

![image](https://user-images.githubusercontent.com/84084372/206623473-7cc6189f-e1a8-4b6d-a500-2e4c81232466.png)


#### 2-1-2) API 활성화

서비스 계정을 사이트 소유자로 추가했으면, API 사용 설정을 해야한다. 우선, 아래 링크를 클릭하여 Google Cloud에 접속해준다.

- **Link**

  <https://console.cloud.google.com/>

검색창에 Google Indexing API를 검색하여, 가장 상단에 노출되는 Indexing API를 클릭해준다.

![image](https://user-images.githubusercontent.com/84084372/206622461-68aa4fc6-8217-43b0-9391-107b9d01154d.png)

`API 사용해 보기`옆의 파란색 버튼을 눌러, API를 사용 설정해주면 된다. 본인은 이미 API를 사용 설정 했기 때문에 화면이 조금 다르게 보일 수 있다.

![image](https://user-images.githubusercontent.com/84084372/206622508-613938f1-1d2e-4e3f-b49e-706e1a79207c.png)

여기까지 따라한다면, API를 사용하기 위한 초기세팅은 모두 끝났다.

<br>

### 2-2) Indexing API 사용

#### 2-2-1) Install oauth2client

우선, 액세스 토큰을 가져오기 위해 oauth2client를 설치해야 한다. pip이나 conda를 사용하여 설치해주도록 하자.

```
>>> pip install oauth2client
```

#### 2-2-2) 스크립트 작성

oauth2client 설치가 끝났으면, 다음과 같은 스크립트를 작성한다. 우선,  

```python
from oauth2client.service_account import ServiceAccountCredentials
import httplib2
import json

############################
# 실제 사용시 이 부분만 수정 #
############################

JSON_KEY_FILE = "C:/Users/USER/Downloads/jekyll-blog-371100-5977418fcddb.json"
URL = "https://on-jungwoan.githb.io/dl/cs231n_52/"
TYPE = "URL_UPDATED"
# TYPE = "URL_DELETED"

#############################
#############################


SCOPES = [ "https://www.googleapis.com/auth/indexing" ]
ENDPOINT = "https://indexing.googleapis.com/v3/urlNotifications:publish"

credentials = ServiceAccountCredentials.from_json_keyfile_name(JSON_KEY_FILE, scopes=SCOPES)

http = credentials.authorize(httplib2.Http())

# Define contents here as a JSON string.
# This example shows a simple update request.
# Other types of requests are described in the next step.

content = """{
  \"url\": \"""" + URL + """\",
  \"type\": \"""" + TYPE + """\"
}"""

response, content = http.request(ENDPOINT, method="POST", body=content)
content_dict = json.loads( content.decode('utf-8') )

if response['status'] != '200':
  print(response['status'], content_dict['error']['message'], sep="\n")
```



## Github Actions

```yaml

name: Python package

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build:

    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.8", "3.9", "3.10"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install flake8 pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

```