---
title:  "[Jekyll]Github io 생성"
excerpt: "Jekyll을 이용한 Github io 생성방법"

categories:
  - TIL
tags:
  - [TIL, jekyll, Github, Githubio]

toc: true
toc_sticky: true
 
date: 2022-07-13
last_modified_at: 2022-07-13
---

# Repository 생성

![image](https://user-images.githubusercontent.com/84084372/178898730-4e84d11b-8975-423e-a5ec-ecf0069401d0.png)
<br>우선, username.github.io의 형식으로 Repository를 생성해준다.<br>
이 때, username이 다르면 사이트가 제대로 생성되지 않는다.<br>

<br>
<br>

# Jekyll 테마 사용

### - 원하는 Jekyll 테마 고르기
----
원하는 Jekyll 테마를 고른다.<br>
본인은 사람들이 가장 많이 사용하는 minimal-mistakes를 선택했다.<br>
처음엔 fork를 하여 사용하였는데 fork하여 commit시, 잔디가 심어지지 않는다는 아주 큰 문제가 있었다!!<br>
그래서 fork Repository를 로컬 환경에 clone한 뒤, 새로운 저장소에 push 해주었다.<br>
커밋 내용이 다 사라진다는 단점이 있었는데, 만든지 얼마 되지 않은 저장소이기 때문에 그냥 push했다.<br>
(git mirror?를 사용하면 커밋 내용도 지킬 수 있다고 한다!)<br>

<br>

### - Jekyll 테마 커스텀
---
테마 커스텀에는 다음과 같은 블로그름 참고하였다.

- 본문 영역 및 글자 크기 조정, 하이퍼링크 밑줄 제거 <br>
https://velog.io/@eona1301/Github-Blog-minimal-mistakes-%EB%B3%B8%EB%AC%B8-%EC%98%81%EC%97%AD-%EB%B0%8F-%EA%B8%80%EC%9E%90-%ED%81%AC%EA%B8%B0 <br>

- 부제목의 소요시간 대신 작성 일자 추가 <br>
https://livlikwav.github.io/etc/blog-revision-1/ <br>

<br>
<br>
<br>

# 최종
![image](https://user-images.githubusercontent.com/84084372/178899861-8cceb0bc-208b-4dec-888d-d8b7aa7d8b7e.png)
