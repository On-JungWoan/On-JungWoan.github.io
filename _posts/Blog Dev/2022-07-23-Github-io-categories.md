---
title:  "[Jekyll]사이드바 카테고리 기능 추가"
excerpt: "Github io의 사이드바 카테고리 기능 추가"

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

<span style='background-color:#ffdce0'><strong>주의!</strong></span>

<span style='background-color:#ffdce0'>
  <strong>
    아래 나와있는 모든 스크립트들은 그대로 복사 붙여넣기 하면 동작하지 않습니다. 
    스크립트를 그대로 푸시하면 페이지 빌드가 제대로 되지 않는 이슈가 있어서 부득이하게 코드를 변형하여 올렸습니다. 
    '{'와 '}'를 모두 '{ %'와 '% }'로 바꾸어주어야 제대로 동작합니다.(공백 없이)
  </strong>
</span>

<br>


## 1. 페이지 생성

`/_pages`에, 위에서 생각해둔 카테고리에 대한 page를 생성하여준다.
편의를 위해 `/_pages` 밑에 `categories/` 디렉토리를 추가하고, 최종적으로 `/_pages/categories/`에 md파일을 생성해주었다. 
이때, md파일은 모든 카테고리에 대하여 생성해준다. 

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/180470639-79db7a0d-2d5d-4b05-995a-f5d07687c9c9.png"></p>

<br>
  
<div align="center">  
  <Strong>[Blog.md]</Strong>
</div>  

  
```
  ---
  title: "Blog dev"
  layout: archive
  permalink: categories/Blog
  author_profile: true
  sidebar_main: true
  ---
  
  { assign posts = site.categories.Blog }
  { for post in posts } { include archive-single.html type=page.entries_layout } { endfor }
```

## 2. 사이드바에 띄우기
`/_include/` 경로 밑에 `nav_list_main` 문서를 새롭게 작성한다. 
이 때, 파일명은 꼭 nav_list_main이 아니어도 된다. 
0번에서 정의한 구조에 맞게 스크립트를 작성하여준다.

<br>

<div align="center">  
  <Strong>[nav_list_main]</Strong>
</div>  

```html
  { assign sum = site.posts | size }
  <nav class="nav__list">
    <input id="ac-toc" name="accordion-toc" type="checkbox" />
    <label for="ac-toc">{{ site.data.ui-text[site.locale].menu_label }}</label>
    <ul class="nav__items" id="category_tag_menu">
        <li>
              📂 <span style="">전체 글 수</style> <span style="">{{sum}}</style> <span style="">개</style> 
        </li>
        <li>
          <span class="nav__sub-title">AI</span>
              <ul>
                  { for category in site.categories }
                      { if category[0] == "ML" }
                          <li><a href="/categories/ML" class="">머신러닝 ({{category[1].size}})</a></li>
                      { endif }
                  { endfor }
              </ul>
              <ul>
                  { for category in site.categories }
                      { if category[0] == "DL" }
                          <li><a href="/categories/DL" class="">딥러닝 ({{category[1].size}})</a></li>
                      { endif }
                  { endfor }
              </ul>
          <span class="nav__sub-title">대외활동</span>
              <ul>
                  { for category in site.categories }
                      { if category[0] == "Internship" }
                          <li><a href="/categories/Internship" class="">인턴 ({{category[1].size}})</a></li>
                      { endif }
                  { endfor }
              </ul>
          <span class="nav__sub-title">etc</span>
              <ul>
                  { for category in site.categories }
                      { if category[0] == "Blog" }
                          <li><a href="/categories/Blog" class="">Blog Dev ({{category[1].size}})</a></li>
                      { endif }
                  { endfor }
              </ul>            
        </li>
    </ul>
  </nav>
```

## 3. sidebar.html 수정
/\_includes/sidebar.html의 내용을 다음과 같이 수정한다.

<br>

<div align="center">  
  <Strong>[sidebar.html]</Strong>
</div>  

```html

{ if page.author_profile or layout.author_profile or page.sidebar }
  <div class="sidebar sticky">
  { if page.author_profile or layout.author_profile }{ include author-profile.html }{ endif }
  { if page.sidebar }
    { for s in page.sidebar }
      { if s.image }
        <img src="{{ s.image | relative_url }}"
             alt="{ if s.image_alt }{{ s.image_alt }}{ endif }">
      { endif }
      { if s.title }<h3>{{ s.title }}</h3>{ endif }
      { if s.text }{{ s.text | markdownify }}{ endif }
      { if s.nav }{ include nav_list nav=s.nav }{ endif }
    { endfor }
    { if page.sidebar.nav }
      { include nav_list nav=page.sidebar.nav }
    { endif }
  { endif }

  { if page.sidebar_main }
    { include nav_list_main }
  { endif }

  </div>
{ endif }

```

div가 끝나기 전에 다음과 같은 script를 추가한 것이다.

```html

  { if page.sidebar_main }
    { include nav_list_main }
  { endif }
  
```  

## 4. \_config.yml / index.html 수정
최상단 경로에 있는 /\_config.yml과 /index.html을 수정한다. 
다음 스크립트와 같이 sidebar_main: true를 추가하여 사이드바가 활성화되게끔 해주면 된다.

<div align="center">  
  <Strong>[_config.yml]</Strong>
</div>  

```

# Defaults
defaults:
  # _posts
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: true
      read_time: true
      comments: true
      share: true
      related: true
      mathjax: true
      sidebar_main: true     # 요거 추가
      
```

<div align="center">  
  <Strong>[index.html]</Strong>
</div>  

```html

layout: home
sidebar_main: true    # 요거 추가
author_profile: true

```

## 5. 최종 결과물

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/180473732-1f6864ec-cba5-4223-a3a9-8d02119de4bc.png"></p>

  
