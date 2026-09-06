---
title:  "[Jekyll]블로그 개선"
excerpt: "상단 네비게이션바 수정 및 tease image 제거"

categories:
  - web-development
tags:
  - [Github, Githubio, jekyll]

published: true

toc: true
toc_sticky: true
 
date: 2022-09-19
last_modified_at: 2022-09-19
til: 'true'
permalink: "/blog/edit_navigation/"
legacy_category: Blog
---

## 1. 상단 네비게이션 바 목록 수정

상단 네비게이션 바에 불필요한 카테고리가 많이 있는 관계로 조금 제거해주기로 하였다.
`/_data/navigation.yml`의 내용을 다음과 같이 수정하여주었다.

<div align="center"><strong>[수정 전]</strong></div>

```yml
main:
  - title: "Home"   # 보여지는 이름 
    url: https://on-jungwoan.github.io/ # 이동하는 url
  - title: "Category"
    url: /categories/
  - title: "Tag"
    url: /tags/
  - title: "Posts"
    url: /year-archive/
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/84084372/190978103-a03d2807-0f36-4d17-afc7-0a046250a32b.png">
</p>

<div align="center"><strong>[수정 후]</strong></div>

```yml
main:
  - title: "Home"   # 보여지는 이름 
    url: https://on-jungwoan.github.io/ # 이동하는 url
  - title: "Category"
    url: /categories-grid/
```
<p align="center">
  <img src="https://user-images.githubusercontent.com/84084372/190978440-101b2cd6-e2a6-4559-87a0-9f4becaa2212.png">
</p>
  
<br>
<br>

## 2. Tease Image 제거

게시물마다 계속 붙어나오는 tease image를 제거해주었다.

<div align="center"><strong>[수정 전]</strong></div>

```yml
"/assets/image/profile_image.jpg" # path of fallback teaser image, e.g. "/assets/images/500x300.png"
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/84084372/190979266-cd8d1461-c4e5-47ab-9ad9-13c749197bb1.png">
</p>

<div align="center"><strong>[수정 후]</strong></div>

```yml
# "/assets/image/profile_image.jpg" # path of fallback teaser image, e.g. "/assets/images/500x300.png"
```
<p align="center">
  <img src="https://user-images.githubusercontent.com/84084372/190979274-9b15701a-415a-4201-b5dd-e1a12286e46e.png">
</p>