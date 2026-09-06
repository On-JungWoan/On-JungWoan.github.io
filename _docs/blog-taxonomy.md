# 블로그 글 분류

카테고리는 `_data/blog_categories.yml`의 9개 key 중 하나만 사용한다.
상위 4개 영역은 메뉴용 그룹이며 post의 `categories`에 추가하지 않는다.

| key | 분류 기준 |
| --- | --- |
| papers | 논문의 아이디어·방법·결과를 설명하는 리뷰 |
| ai-study | AI 개념, 강의 정리, 모델 구현과 실험 |
| model-optimization | 모델 변환, 추론 성능 개선, 배포를 위한 최적화 |
| web-development | 웹 서비스와 Jekyll 블로그 개발 |
| programming-tools | Python 문법과 Git·Docker 등 개발 도구 |
| data-sql | 데이터 처리·분석, SQL |
| projects | 특정 프로젝트의 설계·진행·문제 해결·결과 |
| experiences | 인턴 지원과 활동 기록 |
| life | 일상·여행·음악 등 취미 |

일반적인 TensorRT 사용법은 `model-optimization`, 특정 프로젝트에 적용한 과정은
`projects`에 넣고 `TensorRT` 태그를 단다. 작성 시기나 인턴 여부보다 글의 중심 내용을 기준으로 분류한다.

## 시리즈와 태그

- 프로젝트 연재와 경험별 기록은 `series`에 `_data/blog_series.yml`의 key를 넣는다.
- 시리즈 데이터의 `category`는 `projects` 또는 `experiences`로 지정한다. 각 카테고리는 해당 시리즈만 보여준다.
- 시리즈 페이지는 작성 순서로 보여주며 이전·다음 글도 같은 시리즈 안에서 연결한다.
- 논문은 `_data/blog_topics.yml`의 주제 key를 일반 `tags`에 추가한다. 여러 주제를 지정할 수 있다.
- 연구 주제 페이지는 `papers` 카테고리와 해당 태그에 모두 속하는 글을 보여준다.
- 새 시리즈나 연구 주제를 만들면 데이터와 함께 `_pages/series/` 또는 `_pages/topics/`의
  기존 페이지 형식을 따라 목록 페이지를 추가한다.

```yaml
categories:
  - projects
series: cctv
tags:
  - TensorRT
```

## 기존 링크 유지

2026-09-06 재분류한 공개 글 120개는 기존 주소를 `permalink`에 고정했다.
이 값은 이후 카테고리가 바뀌어도 유지한다. `legacy_category`는 예전 카테고리 페이지에서
기존 글 목록을 제공하는 용도이며, 새 글에는 추가하지 않는다.

파일 이름·폴더와 본문·작성일은 재분류 과정에서 바꾸지 않았다.
공개되지 않은 초안은 이번 일괄 변경 대상에서 제외했으므로, 발행할 때 위 기준에 맞춰 분류한다.
