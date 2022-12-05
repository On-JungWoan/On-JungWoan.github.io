var store = [{
        "title": "[CS231n]Convolutional Neural Networks(초안)",
        "excerpt":"CNN 1. Fully Connected Layer FC Layer는, input 이미지를 1차원으로 쭉 펴서 Weight와 곱해주는 Layer이다. 가령, 32x32x3의 input image가 있다면, FC Layer는 이를 3072x1의 벡터로 쭉 핀 다음 W와 내적을 하여 activation map을 얻는다. 가장 간단하게 생각할 수 있는 layer이긴 하지만, Fully Connected Layer에는 치명적인 단점이 2가지 존재한다. 이미지의 지역적...","categories": ["DL"],
        "tags": ["DL","ICT인턴십"],
        "url": "/dl/cs231n_5/",
        "teaser": null
      },{
        "title": "[ICT 인턴십]2022년 11월 TIL",
        "excerpt":"✍ 학습내용 정리 🕮 TIL &lt;1주차&gt; 느낀점 및 업무내용 : 이번주는 web 퍼블리싱이 아직 덜 되어서 별 작업 없이 일주일을 보냈다. 차장님께 문의드린 결과, web 퍼블리싱이 완료될 때까지 대기라하고만 하셔서 11월 1일에 있는 경진대회 준비를 하였다. 퇴근 후에도 열심히 준비했고 결과는 1등! 11/01 업무지시 업무지시 없음 학습내용 1) [뻐정] 프로젝트...","categories": ["TIL"],
        "tags": ["인턴","ICT인턴십","TIL"],
        "url": "/til/TIL-11/",
        "teaser": null
      },{
        "title": "[CS231n]Training Neural Networks, Part I(초안)",
        "excerpt":"activation functions FC, CNN 등등의 Layer는, 데이터 입력이 들어오면 가중치와 곱하는 연산을 마친 뒤, 활성함수(비선형 연산)를 거치게 된다. 지금부터는 활성 함수의 종류와 장단점에 대하여 소개한다. sigmoid \\[\\sigma(x) = \\frac{1}{1+e^{-x}}\\] 각 입력을 받아서 그 입력을 0~1 사이의 값이 되도록 해준다. 입력값이 크면 출력은 1에 가까울 것이고, 그렇지 않으면 0에 가까울 것이다....","categories": ["DL"],
        "tags": ["DL","ICT인턴십"],
        "url": "/dl/cs231n_6/",
        "teaser": null
      },{
        "title": "[ICT 인턴십]2022년 12월 TIL",
        "excerpt":"✍ 학습내용 정리         🕮 TIL  &lt;1주차&gt;  12/01           업무지시       업무지시 없음            학습내용       1) [CS231n]Training Neural Networks, Part I       12/02           업무지시       업무지시 없음            학습내용       1) [업무 자동화] 행 생성 업무 자동화      ","categories": ["TIL"],
        "tags": ["인턴","ICT인턴십","TIL"],
        "url": "/til/TIL-12/",
        "teaser": null
      },{
        "title": "[업무 자동화] 행 생성 업무 자동화(초안)",
        "excerpt":"[auto_raw_make.py] import os import pandas as pd import numpy as np import random PATH = 'C:/Users/USER/Desktop/자료/excel_50_row_온정완' SAVE_PATH = 'C:/Users/USER/Desktop/자료/result' file_dir_list = [] for _, _, files in os.walk(PATH): for f in files: tmp_df = pd.read_excel(os.path.join(PATH, f)) for row in range(len(tmp_df), 51): row_tmp = [row] for col in range(1, len(tmp_df.columns)): if...","categories": ["toy_project"],
        "tags": ["python","ICT인턴십"],
        "url": "/toy_project/auto_raw/",
        "teaser": null
      }]
