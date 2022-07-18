---
title:  "[Github.io]댓글 기능, Google 노출"
excerpt: "Github io의 댓글 기능을 추가하고 Google에 노출 시키기"

categories:
  - TIL
tags:
  - [TIL, jekyll, Github, Githubio]

toc: true
toc_sticky: true
 
date: 2022-07-19
last_modified_at: 2022-07-19
---
<br>

내 개발 블로그를 구글에 검색해봤는데 아무것도 나오지 않았따. 
그래서 개발 블로그를 구글에 노출시키고 댓글 기능도 좀 추가해보려고 한다.

<br>

## utterances를 사용한 댓글 기능 추가
![image](https://user-images.githubusercontent.com/84084372/179539589-1b54e5a6-3441-4535-bf95-55389139ff76.png)

<br>

댓글 플랫폼 중 **utterances**가 괜찮다고 하길래 해당 댓글 플랫폼을 사용하기로 했다. 
댓글을 달기 위해서는 깃허브 계정이 필요하다고 한다. 
그래도 명색이 개발 블로그인만큼 방문자분들은 개인 github 계정을 가지고 계실 것으로 예상된다. 
댓글에 마크다운도 사용 가능하다고 한다.

<br>

### - **1. 댓글 이슈가 올라올 Repository 생성** <br>
나는 그냥 github.io Repository를 사용하였다. 
댓글이 달리면 해당 Repository에 issue로 올라온다고 한다. 
혹시 댓글 전용 Repository를 만들고 싶으면, 미리 생성하도록 하자. <br>
https://github.com/On-JungWoan/On-JungWoan.github.io/new/main/_posts

<br>


### - **2-1. minimal-mistakes를 사용하는 경우** <br>
minimal-mistakes를 사용한다면, 별도의 과정 없이 그냥 config.yml 파일을 수정해주면 된다. 
수정해야 할 것들은 다음과 같다. <br>

![image](https://user-images.githubusercontent.com/84084372/179542095-812671f7-5280-461a-9a7f-a34a51ba60a9.png) <br>

- repository
  - 위에서 댓글 Issue 가 올라올 곳으로 선택한 그 저장소의 permalink
- provider
  - utterances 를 앞으로 사용할 것이므로 utterances 를 입력
- theme
  - 위에서 설정한 테마 (본인은 gruvbox-dark를 이용)
- issue_term
  - 맵핑키. 웬만하면 default값을 사용하자

<br>

### - **2-1. minimal-mistakes를 사용하지 않는 경우** <br>
https://github.com/apps/utterances
해당 링크를 통해 설치한다.

- repo
  - 위에서 댓글 Issue 가 올라올 곳으로 선택한 그 저장소의 permalink 를 써준다. (github아이디/저장소이름)

- Blog Post - Issue Mapping
  - 댓글 이슈를 댓글 달린 블로그 페이지의 어떤 부분과 매핑을 시킬지 Key를 결정한다.
  - 매핑시키는 것이니만큼 Key 가 달라지면 Value 는 사라질 것이다. 고유하고 수정을 제일 안할 것 같은 pathname을 선택하는 것이 좋을 것 같다.

- Theme
  - utterances 의 테마를 정한다. 난 gruvbox-dark 테마를 선택했다.

- Enable utterances
  - 댓글 구현을 담당하는 html 파일에 이 코드를 그대로 복사하여 원하는 위치에 붙여넣어주면 될 것 같다.
