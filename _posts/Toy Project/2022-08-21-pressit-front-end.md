---
title:  "[Press It!]메인 페이지 Front-End Developing"
excerpt: "Press It! : 문서요약 및 자동번역 서비스"

categories:
  - projects
tags:
  - [Bootstrap, Django, Web, FrontEnd]

published: true

toc: true
toc_sticky: true
 
date: 2022-08-21
last_modified_at: 2022-08-21
permalink: "/toy_project/pressit-front-end/"
legacy_category: toy_project
series: press-it
---

## 0. 메인 페이지 구성

메인 페이지는 크게 'URL 입력', '원문 정보', '요약문', 'contact me'의 4가지 섹션으로 구성하였다. 상단 네이게이션 바 및 scroll 아이콘을 통해 각 섹션간 이동을 할 수 있게 하였으며, 애니메이션을 사용하여 부드럽게 전환되도록 하였다. 섹션명은 기존 템플릿을 따라갔으며, url입력='hero', 원문정보='about', 요약문='resume', contactme='contact'이다.

<br>

## 1. URL 입력 섹션
### 1-1. Web 화면
해당 섹션에서는, 가운데 input 박스에 text의 URL을 입력하면 기존에 정의된 함수로 해당 URL이 넘어가 요약문을 출력하는 섹션으로 넘어가도록 설계하였다.

<div align="center"><strong>[데스크탑 이미지]</strong></div>
<p align="center"><img src="https://user-images.githubusercontent.com/84084372/185746090-6b88c8ab-25f3-4058-8ca4-0b61f6563227.png"></p>

<div align="center"><strong>[모바일 이미지]</strong></div>
<p align="center"><img src="https://user-images.githubusercontent.com/84084372/185746087-58ed0170-db7a-4fdb-a898-769c466acead.png"></p>

<br>

### 1-2. 하단 스크롤 버튼 애니메이션 관련 이슈
해당 섹션 하단 스크롤 버튼에 up-down 애니메이션이 들어가있는데 이상하게 다른 섹션에서는 작동하지 않는다. 해당 애니메이션은 animation css 파일에는 없고 style css 파일에 직접 정의되어 있는 애니메이션인데, 이상하게 적용이 안된다. 그래서 다른 섹션에서는 animation css에 나와있는 다른 애니메이션을 사용하였다.

<div align="center"><strong>[hero 섹션 하단 스크롤 버튼]</strong></div>

```html
<a href="#about" class="btn-scroll scrollto" title="Scroll Down"><i class="bx bx-chevron-down"></i></a>
```

<div align="center"><strong>[hero 섹션 btn-scroll class 옵션]</strong></div>

```css
#hero .btn-scroll {
  transition: 0.4s;
  color: rgba(255, 255, 255, 0.6);
  animation: up-down 1s ease-in-out infinite alternate-reverse both;
}
```

<div align="center"><strong>[개발자 도구에서 확인한 애니메이션]</strong></div>
<p align="center"><img src="https://user-images.githubusercontent.com/84084372/185746081-66a27fac-de1c-4ebb-a921-1c1b3aab616d.png"></p>


<div align="center"><strong>[css에 정의된 애니메이션]</strong></div>

```css
@-webkit-keyframes up-down {
  0% {
    transform: translateY(5px);
  }

  100% {
    transform: translateY(-5px);
  }
}

@keyframes up-down {
  0% {
    transform: translateY(5px);
  }

  100% {
    transform: translateY(-5px);
  }
}
```

<br>

### 1-3. 애니메이션 이슈 차선책

해당 이슈는 별다른 해결 방법이 떠오르지 않아서 일단 다른 섹션에서는 flashing 애니메이션을 사용하였다.

<div align="center"><strong>[about 섹션의 하단 스크롤 버튼]</strong></div>

```css
.about .btn-scroll {
  transition: 0.4s;
  color: rgba(0, 0, 0, 0.6);
  animation: flashing 1.5s infinite;
}
```

<div align="center"><strong>[animation.css]</strong></div>

```css
@-webkit-keyframes flashing
{
    0%
    {
        opacity: 1;
    }
    45%
    {
        opacity: 0;
    }
    90%
    {
        opacity: 1;
    }
}
@keyframes flashing
{
    0%
    {
        opacity: 1;
    }
    45%
    {
        opacity: 0;
    }
    90%
    {
        opacity: 1;
    }
}
```

<br>

## 2. 원문 정보
### 2-1. Web 화면
해당 섹션에서는 URL 입력 섹션에서 입력받은 URL을 바탕으로 원문의 정보를 추출하여 사용자에게 출력한다. 최상단에 원문에 첨부된 대표 이미지를 출력하고, 그 하단에 원문의 URL이 링크된 원문 제목을 출력한다. 번역하고자 하는 언어를 선택한 뒤 스크롤 버튼을 누르면, 요약문 섹션으로 넘어가며 요약문이 생성되고, 사용자가 선택한 언어로 번역된다.

<div align="center"><strong>[데스크탑 이미지]</strong></div>
<p align="center"><img src="https://user-images.githubusercontent.com/84084372/185746099-80dbd40c-a31a-49c0-8cc0-d65b149fd68f.png"></p>


<div align="center"><strong>[모바일 이미지]</strong></div>

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/185746098-730ac461-f619-4a58-8d6a-4113b0ac0caa.png"></p>


<br>

### 2-2. 구조 변경
기존 템플릿의 구조는 하나의 row를 이미지와 text 2개의 컬럼이 나누고 있는 구조를 가지고 있었다. 하지만 원문 제목과 url이 길어질 경우 text가 2줄로 넘어가서 출력되는 것이 싫었다. 따라서 기존의 행 방향으로 text를 늘어뜨리는 방식에서 열 방향으로 늘어뜨리는 방식으로 구조를 변경하였다. 또한, 필요 없는 정보는 과감히 제거하였다.

<div align="center"><strong>[기존 원문정보 페이지]</strong></div>
<p align="center"><img src="https://user-images.githubusercontent.com/84084372/185746097-ae92d74c-f447-4e00-be15-6d338f3af9da.jpg"></p>

<div align="center"><strong>[기존 행방향 구조]</strong></div>

```html
<div class="row">

  <!-- 이미지 영역 -->
  <div class="col">
    이미지
  </div>

  <!-- 기타 글 영역 -->
  <div class="col">
    기타 글
  </div>

</div>
```

<div align="center"><strong>[변경된 열방향 구조]</strong></div>

```html
<!-- 이미지 -->
<div class="row">
</div>

<!-- 원문 제목 -->
<div class="row">
</div>

<!-- 기타 텍스트 -->
<div class="row">
</div>

<!-- 요약문 생성 및 번역 -->
<div class="row">
</div>
```

<br>

### 2-3. 기사 이미지 상한값 조정
또한 섹션의 내용들이 화면을 넘어가지 않게 하기 위해 기사 이미지의 상한 height값을 조정해주었다. 기존 500px에서 350px로 조정해주었다.

<div align="center"><strong>[style.css]</strong></div>

```css
.about .image {
  background: url("../img/test/test.jpg") center center no-repeat;
  background-size: cover;
  min-height: 350px;
}
```

<br>

### 2-4. 하단 스크롤 버튼 생성
다음 요약문 출력 섹션으로 가기위한 스크롤 버튼도 만들어주었다. 애니메이션은 위에서 언급한 바와 같이 flashing을 사용하였다. 1.5초에 걸쳐 애니메이션이 재생되고 이를 무한반복하도록 설정하였다.

<div align="center"><strong>[about 섹션의 하단 이미지 스크롤 버튼]</strong></div>

```html
<a href="#resume" class="btn-scroll" title="Scroll Down"><i class="bx bx-chevron-down"></i></a>
```

<div align="center"><strong>[style.css]</strong></div>

```css
.about .btn-scroll {
  transition: 0.4s;
  color: rgba(0, 0, 0, 0.6);
  animation: flashing 1.5s infinite;
}

.about .btn-scroll i {
  font-size: 48px;
}

.about .btn-scroll:hover {
  color: #ffb727;
}
```

<br>

## 3. 요약문
### 3-1. Web 화면
해당 섹션에서는 요약문과 번역문을 출력한다.

<div align="center"><strong>[데스크탑 이미지]</strong></div>

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/185746129-b47f7aa9-757c-4260-b51d-0d9da114b746.png"></p>

<div align="center"><strong>[모바일 이미지]</strong></div>

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/185746095-6e6199e5-8608-461a-a4d8-52227484ef19.png"></p>


<br>

### 3-2. 하단 스크롤 버튼 설계
하단 스크롤 버튼을 누르면 요약문 섹션으로 다시 올라가게 만들었다. 스크롤 버튼의 이미지를 위를 바라보는 화살표로 변경하고 싶어서 아이콘 관련 css 파일을 찾아보았다. boxicons.css에 해당 아이콘이 정의되어 있었으며, 서버에서 아이콘을 받아오는 방식인 것 같았다. 나는 위로 향하는 아이콘이 필요하기 때문에 해당 아이콘을 찾아서 삽입해주었다.

<div align="center"><strong>[resume 섹션의 하단 스크롤 버튼]</strong></div>

```html
<!-- Next 화살표 -->
<a href="#about" class="btn-scroll scrollto" title="Scroll Down"><i class="bx bx-chevron-up"></i></a>
```

<div align="center"><strong>[boxicons.css]</strong></div>

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/185746083-381dddb9-41cb-438c-a81f-704f1ef68853.png"></p>

<br>

## 4. Contact Me
### 4-1. Web 화면
해당 섹션에서는 나에 대한 정보를 출력한다.

<div align="center"><strong>[데스크탑 이미지]</strong></div>

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/185746085-1e08c55d-d84b-4501-a159-ce034110807a.png"></p>

<div align="center"><strong>[모바일 이미지]</strong></div>

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/185756398-8d34ef24-1968-4a4f-b7df-1d7fe5218356.png"></p>

<br>

### 4-2. social-links 아이콘 변경
Social Profiles의 하단 아이콘은 부트스트랩 아이콘을 지원하는 것 같길래 내 입맛대로 바꾸어보았다.

```html
<div class="social-links">
<a href="https://github.com/on-jungwoan" class="github"><i class="bi bi-github"></i></a>
<a href="https://on-jungwoan.github.io" class="github-io"><i class="bi bi-house-door-fill"></i></a>
<a href="https://instagram.com/dog_woan" class="instagram"><i class="bi bi-instagram"></i></a>
<a href="mailto:slalfpdl16@naver.com" class="linkedin"><i class="bi bi-envelope-fill"></i></a>
</div>
```

<div align="center"><strong>[bootstrap-icons.css]</strong></div>

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/185756584-94e3cbb3-0c62-43d6-a237-201cd6741209.png"></p>

<div align="center"><strong>[부트스트랩 아이콘 홈페이지]</strong></div>

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/185756641-4dfc8271-1dc5-4591-8867-40822e396dc4.png"></p>
