---
title:  "[Github_Io]상단 네비게이션 바 수정"
excerpt: "네비바 수정"

categories:
  - Blog
tags:
  - [Github, Githubio, jekyll]


published: true

toc: true
toc_sticky: true
 
date: 2022-11-04
last_modified_at: 2022-11-04
til: 'true'
---

## navigation.yaml 수정

나는 기존에 사용하던 카테고리가 있었기 때문에, 해당 카테고리를 그대로 구현해주었다. 이 때, AI와 Platform Dev 섹션은 Study Note로 묶어주었다. sublink는 최대 2단계까지만 지원한다고 한다.

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/199757662-7074cb35-28d9-49e7-bfb9-d9481b9e293e.png" style="border: 2px solid black"></p>

<div align="center"><strong>[navigation.yaml]</strong></div>

```yaml
main:
  - title: "PROJECTS"
    sublinks:
      - title: "뻐정 : 버스정보종합시스템"
        url: /categories/bbeojung    
      - title: "실시간 쓰레기 무단투기 탐지"
        url: /categories/cctv
      - title: "canvas 개발"
        url: /categories/canvas
      - title: "Toy Project"
        url: /categories/toy_project

  - title: "STUDY NOTE"
    sublinks:
      - title : 'AI'
        sublinks:
          - title: "Computer Vision"
            url: /categories/DL
          - title: "Machine Learning"
            url: /categories/ML
      - title : 'PLATFORM DEV.'
        sublinks:
          - title: "Blog Dev"
            url: /categories/Blog
          - title: "Web"
            url: /categories/web
          - title: "Tools"
            url: /categories/tools
        
  - title: "대외활동"
    sublinks:
      - title: "2022 하반기 ICT 인턴십"
        url: /categories/internship
        
  - title: "ETC"  
    sublinks:
      - title: "Python 개념"
        url: /categories/python
      - title: "화성학"
        url: /categories/harmonics
      - title: "etc"
        url: /categories/etc

  - title: "TIL"    
    url: /TIL/
    
```

<br>
<br>

## masthead.html 수정

masthead에 navigation 정보를 불러오는 코드를 넣어준다.

![image](https://user-images.githubusercontent.com/84084372/199758613-c1d0d41d-73ef-48a7-a0b5-b8c6623f11d2.png)

이 때, 하위 sublink가 있으면 for문을 한 번 더 돌게 하였고 그렇지 않으면 바로 sublink 정보를 가져오도록 하였다.

![image](https://user-images.githubusercontent.com/84084372/199758848-0c930ff9-3ed4-462a-a419-1819e1766f24.png)

<br>
<br>

## navigation.scss 수정
.greedy-nav{}안에 다음 style을 넣어주면 된다.

```css
  // to be shared by both hidden link and visible link
  .dropdown {
    // float: left;
    width: auto;
    box-sizing: border-box;
  }

  .dropdown-content {
    display: none;
    position: absolute;
    background-color: darken($background-color, 3%);
    // min-width: 160px;
    box-shadow: $box-shadow;
    z-index: 1;
    box-sizing: border-box;
  }

  .dropdown:hover .dropdown-content {
    display: block;
    // font-size: 17px;
  }

  .dropdown-item {
    padding: 10px;
    font-size: 15px;
  }


  .visible-links {
    display: -webkit-box;
    display: -ms-flexbox;
    display: flex;
    -webkit-box-pack: end;
    -ms-flex-pack: end;
    justify-content: flex-end;
    -webkit-box-flex: 1;
    -ms-flex: 1;
    flex: 1;
    overflow: hidden;

    li {
      -webkit-box-flex: 0;
      -ms-flex: none;
      flex: none;
      display: list-item;
      text-align: -webkit-match-parent;
    }

    a {
      position: relative;

      &:before {
        content: "";
        position: absolute;
        left: 0;
        bottom: 0;
        height: 4px;
        background: $primary-color;
        width: 100%;
        -webkit-transition: $global-transition;
        transition: $global-transition;
        -webkit-transform: scaleX(0) translate3d(0, 0, 0);
        transform: scaleX(0) translate3d(0, 0, 0); // hide
      }

      &:hover:before {
        -webkit-transform: scaleX(1);
        -ms-transform: scaleX(1);
        transform: scaleX(1); // reveal
      }
    }
  }


  .hidden-links {
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 15px;
    padding: 5px;
    border: 1px solid $border-color;
    border-radius: $border-radius;
    background: $background-color;
    -webkit-box-shadow: 0 2px 4px 0 rgba(#000, 0.16),
      0 2px 10px 0 rgba(#000, 0.12);
    box-shadow: 0 2px 4px 0 rgba(#000, 0.16), 0 2px 10px 0 rgba(#000, 0.12);

    &.hidden {
      display: none;
    }

    a {
      margin: 0;
      padding: 10px 20px;
      font-size: $type-size-5;

      &:hover {
        color: $masthead-link-color-hover;
        background: $navicon-link-color-hover;
      }
    }

    &:before {
      content: "";
      position: absolute;
      top: -11px;
      right: 10px;
      width: 0;
      border-style: solid;
      border-width: 0 10px 10px;
      border-color: $border-color transparent;
      display: block;
      z-index: 0;
    }

    &:after {
      content: "";
      position: absolute;
      top: -10px;
      right: 10px;
      width: 0;
      border-style: solid;
      border-width: 0 10px 10px;
      border-color: $background-color transparent;
      display: block;
      z-index: 1;
    }

    li {
      display: block;
      border-bottom: 1px solid $border-color;

      &:last-child {
        border-bottom: none;
      }
    }
  }//hidden link
```

위 css에서 dropdown-title는 굵게 표시하였고, 마우스 호버링 이벤트는 dropdown-item에만 들어가도록 수정하였다.

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/199765353-e752bb34-63ea-41be-ac8a-07a511b8d6af.png"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/199765392-a64bf886-7028-4cb7-a900-0aee8cd8924c.png"></p>


<br>
<br>

## Dropdown 아이콘이 안불러와지는 경우

다음과 같이 dropdown 아이콘이 안불러와지는 경우가 있다. 

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/199761237-3dfa421b-550c-4818-87c3-1b2c81d01c6f.png" style="border: 2px solid black"></p>

<br>

masthead.html 최상단에 icon.html을 불러와주면 해결된다.

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/199764450-eb4dd6fc-9c09-4d58-bbd9-769d2dfa8620.png"></p>


<p align="center"><img src="https://user-images.githubusercontent.com/84084372/199761635-ad29b45c-cb82-4417-9515-c149b4d28fe0.png" style="border: 2px solid black"></p>