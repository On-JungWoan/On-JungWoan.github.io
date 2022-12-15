var store = [{
        "title": "[ICT 인턴십]2022년 11월 TIL",
        "excerpt":"✍ 학습내용 정리 Computer Vision [CS231n]Image Classification [CS231n]Loss Functions and Optimization [CS231n]Backpropagation and Neural Networks [CS231n]Convolutional Neural Networks Web [JQuery]AJAX를 활용한 비동기적 데이터 교환 [HTML, JavaScript]Input 태그 유효성 검사 [CSS] 폰트 관련 속성 정리 [Django] Views.py 분리 [Django]secret 관리 [Django] 이미 존재하는 DB 연동 by inspectdb [Django]DRF(Django Rest Framework) 듀토리얼...","categories": ["TIL"],
        "tags": ["인턴","ICT인턴십","TIL"],
        "url": "/til/TIL-11/",
        "teaser": null
      },{
        "title": "[CS231n]Training Neural Networks, Part I",
        "excerpt":"Ⅰ. activation functions FC, CNN 등등의 Layer는, 데이터 입력이 들어오면 가중치와 곱하는 연산을 마친 뒤, 활성함수(비선형 연산)를 거치게 된다. 지금부터는 활성 함수의 종류와 장단점에 대하여 소개한다. 1. sigmoid \\[\\sigma(x) = \\frac{1}{1+e^{-x}}\\] 각 입력을 받아서 그 입력을 0~1 사이의 값이 되도록 해준다. 입력값이 크면 출력은 1에 가까울 것이고, 그렇지 않으면 0에...","categories": ["DL"],
        "tags": ["DL","ICT인턴십"],
        "url": "/dl/cs231n_6/",
        "teaser": null
      },{
        "title": "[ICT 인턴십]2022년 12월 TIL",
        "excerpt":"✍ 학습내용 정리 🕮 TIL &lt;1주차&gt; 12/01 업무지시 업무지시 없음 학습내용 1) [CS231n]Training Neural Networks, Part I 12/02 업무지시 업무지시 없음 학습내용 1) [업무 자동화] 행 생성 업무 자동화 &lt;2주차&gt; 12/06 업무지시 업무지시 없음 학습내용 1) [CS231n]Training Neural Networks, Part Ⅱ 12/09 업무지시 업무지시 없음 학습내용 1) [Google Indexing API,...","categories": ["TIL"],
        "tags": ["인턴","ICT인턴십","TIL"],
        "url": "/til/TIL-12/",
        "teaser": null
      },{
        "title": "[업무 자동화] 행 생성 업무 자동화(초안)",
        "excerpt":"[auto_raw_make.py] import os import pandas as pd import numpy as np import random PATH = 'C:/Users/USER/Desktop/자료/excel_50_row_온정완' SAVE_PATH = 'C:/Users/USER/Desktop/자료/result' file_dir_list = [] for _, _, files in os.walk(PATH): for f in files: tmp_df = pd.read_excel(os.path.join(PATH, f)) for row in range(len(tmp_df), 51): row_tmp = [row] for col in range(1, len(tmp_df.columns)): if...","categories": ["toy_project"],
        "tags": ["python","ICT인턴십"],
        "url": "/toy_project/auto_raw/",
        "teaser": null
      },{
        "title": "[CS231n]Training Neural Networks, Part Ⅱ",
        "excerpt":"Optimization Normalization 시키지 않고, zero-centered 하지 않은 데이터는 학습시키기 어렵다. 아래는 Normalization 하기 전과 후의 데이터를 classifier로 분류하는 과정을 도식화 한 것이다. 위 그림에서도 알 수 있듯이, Normalization이 되지 않은 그래프는 Classifier가 조금만 움직여도 제대로 분류가 되지 않는다. 즉, Loss가 파라미터의 변화에 매우 민감하기 때문에 동일한 함수를 써도 학습시키기 어렵다....","categories": ["DL"],
        "tags": ["DL","ICT인턴십"],
        "url": "/dl/cs231n_7/",
        "teaser": null
      },{
        "title": "[CS231n]CNN Architectures",
        "excerpt":"1. LeNet (LeCun, 1998) 1-1) 구조 LeNet은 industry에 아주 성공적으로 적용된 최초의 Convolution Network이다. [CONV - POOL] 구조가 2번 반복되고, filter는 strdie=1의 5x5 filter를 사용하며 끝단에는 2개의 FC Layer를 쌓았다. 매우 간단한 구조이지만, 꽤 우수한 성능을 보이는 CNN의 조상이다. 2. AlexNet (Krizhevsky, 2012) 최초의 Larg scale CNN이며, ImageNet Classification Task에서...","categories": ["DL"],
        "tags": ["DL","ICT인턴십"],
        "url": "/dl/cs231n_9/",
        "teaser": null
      },{
        "title": "[Google Indexing API, Github Actions] 구글 서치 콘솔 색인 생성 자동화 하는 법(초안)",
        "excerpt":"1. 기존의 페이지 색인 생성 방식 문제점 기존에는 페이지 색인을 생성하기 위해, Google Search Console에 들어가서 일일이 URL을 입력하고, 색인 생성 요청을 누른 뒤, 1~2분을 기다려야 했다. 매번 포스팅을 작성할 때마다 이런 번거로운 일을 하기도 귀찮고, 한동안 까먹고 밀리기라도 하면, 매우 끔찍하다… 블로그 조회수와도 직결되는 중요한 작업이기 때문에 안할수도 없어서,...","categories": ["Blog"],
        "tags": ["Githubio","jekyll","html"],
        "url": "/blog/auto_indexing/",
        "teaser": null
      },{
        "title": "[CS231n]Recurrent Neural Networks(초안)",
        "excerpt":"1. Process Sequences 1-1) Vanilla Neural Network One to One 지금까지 공부한 Vanilla Neural Network 아키텍쳐들은, 이미지(or 벡터) 하나를 입력으로 받아 Hidden Layer를 거쳐 하나의 출력을 내보낸다. 그러나 세상의 다양한 문제를 해결하기 위해서는, 입력과 출력의 개수를 유동적으로 받아들일 수 있는 모델이 필요하다. 이렇게 유동적인 입/출력을 처리해줄 수 있는 네트워크가 바로...","categories": ["DL"],
        "tags": ["DL","ICT인턴십"],
        "url": "/dl/cs231n_10/",
        "teaser": null
      }]
