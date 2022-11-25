var store = [{
        "title": "[Jekyll]포스트에 특정 문구 고정(초안)",
        "excerpt":"우선, 우리가 게시물을 작성하면 게시물 -&gt; single.html -&gt; default.html의 순서로 layout이 상속되어 화면에 출력된다. 여기서, 게시물의 직접적인 요소가 담겨있는 layout은 single.html이다. 따라서 해당 템플릿의 내용을 적절히 수정해주면 포스팅 형식을 바꿀 수 있다. ㅁ single.html의 중간의 {{content}}가 포스팅 내용이 들어가는 영역이다. 따라서 해당 부분 전후로 원하는 문구를 넣으면, 포스팅에 해당 문구가...","categories": ["Blog"],
        "tags": ["Githubio","jekyll","html"],
        "url": "/blog/post_custom/",
        "teaser": null
      },{
        "title": "[CS231n]Image Classification",
        "excerpt":"구현 코드 Lecture_2.ipynb Image Classification? Image Classification이란, Input image를 받아 미리 정해놓은 category 집합 중 어디에 속하는지를 알아맞추는 Computer Vision 분야이다. 이 과정은 사람에게는 매우 쉽지만 기계에게는 어려운 일이다. 왜냐하면 컴퓨터는 이미지를 거대한 숫자집합으로만 인식하기 때문이다(semantic gap). 이미지에 아주 미묘한 변화만 줘도 픽셀 값들은 모조리 달라지며, 알고리즘은 이런것들(조명, 화각, 객체...","categories": ["DL"],
        "tags": ["DL","ICT인턴십"],
        "url": "/dl/cs231n_2/",
        "teaser": null
      },{
        "title": "[Jekyll]빌드 속도 최적화",
        "excerpt":"1. 최적화 방법 Hugo 대비 Jekyll의 가장 큰 단점은 바로 빌드 시간이 매우 느리다는 것이다. 포스팅의 개수가 늘어날수록 빌드 시간은 매우 느려져서, 혹자의 경우 빌드 시간이 10분을 초과하는 경우도 있다고 한다. 따라서 해당 포스팅에서는 Jekyll 빌드 시간을 최적화하는 방법에 대해 소개한다. 1-1) liquid-c liquid-c를 설치하면, liquid문을 C로 처리하면서 빌드 속도를...","categories": ["Blog"],
        "tags": ["Githubio","jekyll","html"],
        "url": "/blog/optim_jekyll/",
        "teaser": null
      },{
        "title": "[CS231n]Loss Functions and Optimization",
        "excerpt":"가중치 W구하는 법 가중치 W를 구하기 위해서는 해당 가중치가 좋은지 나쁜지를 정량화 할 방법이 필요하다. 따라서 가중치의 성능을 파악하기 위해 도입된 것이 바로 손실함수이다. 행렬 W가 될 수 있는 모든 가능의 수 중에서 손실함수가 가장 적은 W를 찾는 것이 가중치 W를 구하는 방법이며, 이를 최적화 과정(optimization)이라 한다. Loss Loss는 Data...","categories": ["DL"],
        "tags": ["DL","ICT인턴십"],
        "url": "/dl/cs231n_3/",
        "teaser": null
      },{
        "title": "[ICT 인턴십]2022년 12월 TIL",
        "excerpt":" ","categories": ["TIL"],
        "tags": ["인턴","ICT인턴십","TIL"],
        "url": "/til/TIL-12/",
        "teaser": null
      }]
