---
title:  "[Jekyll]블로그 테마 변경 및 커스텀"
excerpt: "테마 커스텀"

categories:
  - web-development
tags:
  - [Github, Githubio, jekyll]

published: true

toc: true
toc_sticky: true
 
date: 2022-09-15
last_modified_at: 2022-09-15
til: 'true'
permalink: "/blog/change_theme/"
legacy_category: Blog
---

## 테마 변경

테마를 기존 "dirt" 테마에서 "contrast" 테마로 변경하였는데, 마음에 들지 않는 부분이 있었다.
링크 텍스트 색이 파랗게 되어있는 것과, 선택 영역이 빨간색인 것들을 수정해주었다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/84084372/190315585-576c7957-de2e-4892-99af-f4510fc3f167.png">
</p>
  
<p align="center">  
  <img src="https://user-images.githubusercontent.com/84084372/190315610-a729a628-2e6d-4857-b5ba-475be038a259.png">
</p>
  
<p align="center">  
  <img src="https://user-images.githubusercontent.com/84084372/190315636-327ea54a-960f-43f7-ace6-a6f4783e3a17.png">
</p>

## css 수정

<div align="center"><strong>[경로]</strong></div>

```
On-JungWoan.github.io/_sass/minimal-mistakes/skins/_contrast.scss
```

<div align="center"><strong>[변경 전]</strong></div>

```css
$primary-color: #ff0000 !default;
$link-color: #0000ff !default;
```

<div align="center"><strong>[변경 후]</strong></div>

```css
$primary-color: #000000 !default;
$link-color: #340000 !default;
```
  
## base.scss 수정

링크에 밑줄이 없으니 텍스트와 구분이 잘 안되어서 a 태그에 밑줄 다시 생성

```css
a {
  // text-decoration: none;  삭제한 코드
   
  &:focus {
    @extend %tab-focus;
  }
```  
  
<p align="center">  
  <img src="https://user-images.githubusercontent.com/84084372/190319331-8b137de1-5797-47bc-8828-e1d081cb4559.png">
</p>  

<p align="center">  
  <img src="https://user-images.githubusercontent.com/84084372/190319346-788872f0-7f8e-4c0b-a1ef-5c6eb05c8596.png">
</p>  
