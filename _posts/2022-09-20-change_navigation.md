---
title:  "[Github_Io]상단 네비게이션바 수정"
excerpt: "상단 네비게이션 바 항목 수정 및 TIL 추가"

categories:
  - Blog
tags:
  - [Github, Githubio, jekyll]


published: true

toc: true
toc_sticky: true
 
date: 2022-09-20
last_modified_at: 2022-09-20
---

## 1. 홈 외의 다른 page에 사이드바 추가

홈 외의 다른 page에서 사이드바가 나오지 않는 문제점을 발견하여 해결하였다.

<div align="center"><strong>[변경전]</strong></div>

<p align="center">
  <img src="https://user-images.githubusercontent.com/84084372/191176919-f2ded323-79f8-4ba3-9026-3f7caada5c59.png" style="border: 2px solid black">
</p>

<br>

`On-JungWoan.github.io/_pages/category-archive-grid.md`의 일부이다. sidebar_main 옵션을 true로 주었다.

```markdown
---
title: "Posts by Category (grid view)"
layout: categories
permalink: /categories-grid/
entries_layout: grid
author_profile: true
sidebar_main: true
---
```
<div align="center"><strong>[변경후]</strong></div>

<p align="center">
  <img src="https://user-images.githubusercontent.com/84084372/191176938-83e97eae-fc76-40a0-89ff-f66d79bead61.png" style="border: 2px solid black">
</p>

<br>

## 2. TIL 페이지 생성
개발 블로그에 매일 포스팅한 내용들이 자동으로 TIL 페이지로 분류되게 하고, 따로 모아볼 수 있게 하고자 하였다. 
따라서 TIL 페이지를 생성하여 상단 네비게이션 바에 추가해주었다. 
다음은 새롭게 생성한 `On-JungWoan.github.io/_pages/TIL.md`의 내용이다. 
포스트를 작성할 때, tag에 TIL을 입력하면 자동으로 TIL 페이지에 분류된다.

```markdown
---
title: "TIL 내용 모음"
layout: archive
permalink: /TIL/
author_profile: true
sidebar_main: true
---

\{\% assign posts = site.tags.TIL \%\}
\{\% for post in posts \%\} \{\% include archive-single.html type=page.entries_layout \%\} \{\% endfor \%\}
```

위에서 만든 페이지를 `On-JungWoan.github.io/_data/navigation.yml`에 등록해주었다.

```markdown
# main links
main:
  - title: "Home"   # 보여지는 이름 
    url: https://on-jungwoan.github.io/ # 이동하는 url
  - title: "Category"
    url: /categories-grid/
  - title: "TIL"    
    url: /TIL/
```

기존 포스트의 tag를 변경하여 테스트 해본 결과 잘 작동하는 것을 확인할 수 있었다.

<br>

```markdown
---
title:  "[ICT 인턴십]9월 TIL"
excerpt: "회사에서 배운 내용들 정리"

categories:
  - Internship
tags:
  - [인턴, ICT인턴십, TIL]
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/84084372/191178629-e6358cfb-5815-4473-9e9a-f0727150e1c0.png" style="border: 2px solid black">
</p>

<br>

<div align='center'>
  <a href="https://on-jungwoan.github.io/internship/Itern-TIL/#0920">
    본문으로 돌아가기
  </a>
</div>
