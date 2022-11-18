---
title:  "user정보 DB 구조파악"
excerpt: "canvas 개발"

categories:
  - canvas
tags:
  - [postgreSQL, DataGrip, 유클리드소프트]

toc: true
toc_sticky: true
 
date: 2022-10-28
last_modified_at: 2022-10-28  
til: 'true'
---

우선, DB 구조를 파악하기 위해 웹에서 스키마를 그려볼 수 있는 사이트를 사용하였다. 아래 사이트에 접속하면 무료로 DB 스키마를 그려볼 수 있다. 디자인도 깔끔하고 사용법도 쉬워서 간단하게 그려보기 좋은 것 같다.

- **사이트 주소**

  [WWW SQL Designer](https://ondras.zarovi.cz/sql/demo/)

아래는 해당 사이트에서 직접 그려본 실제 DB 스키마의 일부이다.

![image](https://user-images.githubusercontent.com/84084372/198542945-80677db2-8b80-4e4e-a43b-5ff33d9405e1.png)

<br>
<br>

## 테이블 설명
### users

회원을 생성할 수 있는 방법은 총 2가지가 있다.

1. Site Admin이 직접 사용자 추가
2. Course에서 사용자 직접 추가

자세한 사항은 이전 포스팅을 참조하면 된다.

- **Refer.**

  [프로젝트 개요 정리](https://on-jungwoan.github.io/canvas/setting_proj/#%ED%95%B4%EC%95%BC%ED%95%A0-%EC%9D%BC)

  
우선, 1번과 2번 방법으로 생성한 사용자 모두 users 테이블에 저장되는 것을 확인하였다. 생성 방법에 상관없이 그냥 모든 유저에 대한 정보가 users에 저장되는 것 같았다. 또한 users 중에서 Site Admin이 생성한 user의 계정 정보가 account에 저장되는 것을 확인하였다. 자세한 사항은 디자인이 넘어오면 다시 확인해봐야겠다.