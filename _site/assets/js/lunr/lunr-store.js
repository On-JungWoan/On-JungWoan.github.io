var store = [{
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
        "title": "[Google search console] 구글 서치 콘솔 색인 생성 자동화 하는 법",
        "excerpt":"기존의 페이지 색인 생성 방식 문제점 기존에는 페이지 색인을 생성하기 위해, Google Search Console에 들어가서 일일이 URL을 입력하고, 색인 생성 요청을 누른 뒤, 1~2분을 기다려야 했다. 매번 포스팅을 작성할 때마다 이런 번거로운 일을 하기도 귀찮고, 한동안 까먹고 밀리기라도 하면, 매우 끔찍하다… 블로그 조회수와도 직결되는 중요한 작업이기 때문에 안할수도 없어서, 이...","categories": ["Blog"],
        "tags": ["Githubio","jekyll","html"],
        "url": "/blog/auto_indexing/",
        "teaser": null
      }]
