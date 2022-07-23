---
title:  "[Github_Io]사이드바 카테고리 기능 추가"
excerpt: "Github io의 댓글 기능을 추가"

categories:
  - Blog
tags:
  - [jekyll, Github, Githubio]

published: true

toc: true
toc_sticky: true
 
date: 2022-07-23
last_modified_at: 2022-07-23
---

<br>

해당 포스팅은 "공부하는 식빵맘"님의 포스트를 참조하여 만들었습니다.<br>
[출처] : https://ansohxxn.github.io/blog/category/

<br>

## 0. 만들고자 하는 카테고리 정의
우선 만들고자 하는 카테고리를 정의하여준다. 
나는 다음과 같은 구조를 가지는 카테고리를 만들고자 하였다.

- AI
  - ML
  - DL
- 대외활동
  - 인턴
- etc
  - Blog Dev


AI, 대외활동, etc는 대분류를 위한 span값이고 ML, DL, 인턴, Blog Dev가 카테고리이다. 
해당 카테고리가 등록되어있는 글을 포스팅하면 자동으로 위의 카테고리에 맞춰서 글이 분류가 된다.

<br>

## 1. 페이지 생성

/_pages에, 위에서 생각해둔 카테고리에 대한 page를 생성하여준다.
편의를 위해 /_pages 밑에 categories/ 디렉토리를 추가하고, 최종적으로 /_pages/categories/에 md파일을 생성해주었다. 
이때, md파일은 모든 카테고리에 대하여 생성해준다. 

<p align="center"><image src="https://user-images.githubusercontent.com/84084372/180470639-79db7a0d-2d5d-4b05-995a-f5d07687c9c9.png">

<br>
  
   
<image src="https://user-images.githubusercontent.com/84084372/180594971-72bd6b5a-7294-4f34-82e8-1cf32e5d0551.png">

<div align="center">  
  <Strong>[Blog.md]</Strong>
</div> 

<br>

## 2. 사이드바에 띄우기

/_include/ 경로 밑에 nav_list_main 문서를 새롭게 작성한다. 
이 때, 파일명은 꼭 nav_list_main이 아니어도 된다. 
0번에서 정의한 구조에 맞게 스크립트를 작성하여준다.

<br>

<image src="https://user-images.githubusercontent.com/84084372/180594853-d60a5f19-01fe-4dcd-b84d-68424d907c82.png">

<div align="center">  
  <Strong>[nav_list_main]</Strong>
</div>  

<br>

## 3. sidebar.html 수정

/_includes/sidebar.html의 내용을 다음과 같이 수정한다.

<br>

<image src="https://user-images.githubusercontent.com/84084372/180595073-1299310b-eb2f-4eab-b186-c820966938fa.png">

<div align="center">  
  <Strong>[sidebar.html]</Strong>
</div>  

<br>

div가 끝나기 전에 다음과 같은 script를 추가한 것이다.

<image src="https://user-images.githubusercontent.com/84084372/180595120-413529c7-88ed-4a14-a2ea-8314b1f6c21b.png">
  
<br>

## 4. _config.yml / index.html 수정

최상단 경로에 있는 /_config.yml과 /index.html을 수정한다. 
다음 스크립트와 같이 sidebar_main: true를 추가하여 사이드바가 활성화되게끔 해주면 된다.

<image src="https://user-images.githubusercontent.com/84084372/180595200-a67057dd-4c09-45e6-83cc-05422a3ded38.png">

<div align="center">  
  <Strong>[_config.yml]</Strong>
</div>  

<image src="https://user-images.githubusercontent.com/84084372/180595270-a914dd7a-10b4-40f3-9efb-97afd4ae05b1.png">

<div align="center">  
  <Strong>[index.html]</Strong>
</div>

<br>

## 5. 최종 결과물

<p align="center"><image src="https://user-images.githubusercontent.com/84084372/180473732-1f6864ec-cba5-4223-a3a9-8d02119de4bc.png">
