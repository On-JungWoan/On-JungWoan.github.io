var store = [{
        "title": "[ICT 인턴십]2022년 12월 TIL",
        "excerpt":"✍ 학습내용 정리         🕮 TIL  &lt;1주차&gt;  12/01           업무지시       업무지시 없음            학습내용       1) [CS231n]Training Neural Networks, Part I       12/02           업무지시       업무지시 없음            학습내용       1) [업무 자동화] 행 생성 업무 자동화           &lt;2주차&gt;  느낀점 및 업무내용 :   12/06           업무지시       업무지시 없음            학습내용       1) [CS231n]Training Neural Networks, Part Ⅱ      ","categories": ["TIL"],
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
        "title": "[CS231n]Training Neural Networks, Part Ⅱ(초안)",
        "excerpt":"Optimization Normalization 시키지 않고, zero-centered 하지 않은 데이터는 학습시키기 어렵다. 아래는 Normalization 하기 전과 후의 데이터를 classifier로 분류하는 과정을 도식화 한 것이다. 위 그림에서도 알 수 있듯이, Normalization이 되지 않은 그래프는 Classifier가 조금만 움직여도 제대로 분류가 되지 않는다. 즉, Loss가 파라미터의 변화에 매우 민감하기 때문에 동일한 함수를 써도 학습시키기 어렵다....","categories": ["DL"],
        "tags": ["DL","ICT인턴십"],
        "url": "/dl/cs231n_7/",
        "teaser": null
      }]
