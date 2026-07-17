# 저장소 작업 지침

## 프로젝트 개요

- GitHub Pages에 배포하는 Jekyll 개인 개발 블로그다. Minimal Mistakes 원격 테마를 사용한다.
- `index.html`은 영문 연구자 포트폴리오, `blog.html`과 `_posts/`는 주로 한국어 블로그다.
- `projects/<slug>/`에는 독립형 연구 프로젝트 페이지가 있다.
- 글·포트폴리오·프로젝트 페이지의 일상적인 수정에는 `$maintain-development-blog`를 사용한다.
- 구조 진단, 중복·미사용 코드 정리, 파일 분리와 같은 리팩터링에는 `$refactor-development-blog`를 사용한다.

## 변경 원칙

- 요청과 가장 가까운 기존 파일을 먼저 확인하고 그 구조와 표현을 따른다.
- 명시적인 리팩터링 요청이 아니면 요청한 범위만 수정한다. 대규모 재정렬, 무관한 문구 교정, 의존성 추가는 피한다.
- 학력, 경력, 수상, 논문 상태, 날짜, URL 같은 개인·연구 정보를 추측해서 만들지 않는다.
- 기존 페이지의 언어를 유지한다. 기본적으로 포트폴리오와 연구 페이지는 영어, 블로그 글은 한국어다.
- Jekyll/Liquid 템플릿에서는 가능하면 `relative_url` 필터를 사용하고, 독립형 정적 페이지는 주변의 경로 방식을 따른다.
- 화면 변경 시 모바일 레이아웃, 키보드 포커스, 이미지 대체 텍스트, `prefers-reduced-motion`을 함께 고려한다.

## 경로별 주의사항

- 새 글은 `_posts/<분류>/YYYY-MM-DD-slug.md`에 UTF-8로 작성하고, 유사한 최신 글의 frontmatter를 기준으로 필요한 항목만 둔다.
- 글 이미지는 보통 `assets/images/posts/<slug>/`, 논문·프로젝트 자료는 `assets/paper/`에 둔다.
- `_site/`는 빌드 결과물이므로 직접 수정하거나 커밋하지 않는다.
- `assets/js/main.min.js`는 생성 파일이다. `assets/js/_main.js` 등을 바꾼 경우에만 `npm run build:js`로 다시 만든다.
- `projects/multi-thumbs/viewer/viser-client/assets/`의 해시 파일은 외부 빌드 결과물이므로 명시적 요청 없이 손대지 않는다.
- `auto_indexing.py`는 자격 증명을 사용해 외부 Google API를 호출하므로 일반 검증 과정에서 실행하지 않는다.

## 검증

- 일반 변경: `bundle exec jekyll build`
- JavaScript 소스 변경: `npm run build:js` 후 Jekyll 빌드
- 화면 변경: 가능하면 로컬 미리보기에서 데스크톱과 모바일 폭을 확인한다.
- 마무리: `git diff --check`와 `git status --short`로 불필요한 변경 및 생성 파일 포함 여부를 확인한다.

검증을 실행하지 못했거나 기존 오류가 있으면 결과에 그 이유를 명시한다.
