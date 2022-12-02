var store = [{
        "title": "[CS231n]Convolutional Neural Networks(초안)",
        "excerpt":"CNN NN : 선형 레이어를 쌓고 그 사이에 비선형 레이어를 추가 -&gt; Mode 문제 해결 how? 자동차를 올바르게 분류하기 위해 중간 단계 템플릿 학습: 노란차, 빨간차 등 그리고 이 템플릿들을 결합해서 최종 클래스 스코어 계산 CNN convolutional layer? 기본적으로 공간적 구조를 유지한다 - perceptron : wx+b와 유사한 함수 사용, but...","categories": ["DL"],
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
        "excerpt":"Mini-batch SGD Mini-batch SGD에 대해 복습하기 NN의 학습 필요한 기본 설정 활성함수 선택 데이터 전처리 가중치 초기화 Regularization gradient checking training dynamics part 1. activation functions 지난시간에 봤던 layer : 데이터 입력이 들어오면 가중치와 곱합 FC, CNN 등등 -&gt; 그리고 나서 활성함수 (비선형 연산)을 거치게 됨. sigmoid 1/(1+…) : 각...","categories": ["DL"],
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
