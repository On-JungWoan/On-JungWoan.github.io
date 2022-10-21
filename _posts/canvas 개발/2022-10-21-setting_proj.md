---
title:  "프로젝트 개요 정리"
excerpt: "canvas 개발"

categories:
  - canvas
tags:
  - [postgreSQL, Docker, DataGrip, 유클리드소프트]

---

## 프로젝트 개요

![image](https://user-images.githubusercontent.com/84084372/197138087-c61eaf62-922a-449d-8c95-db5f4f87755e.png)

회사 내부 학습관리시스템(LMS)의 로그인 창이다. 현재 회사 내부 LMS 기술은 [INSTRUCTURE](https://www.instructure.com/)라는 곳에서 가져왔다고 한다. 거의 기술 이전이 완료된 상태이지만, 아직 로그인 창은 제대로 구현이 되어있지 않은 상태라고 한다. 따라서 일단 Django 템플릿으로 로그인 canvas를 개발하는 것이 주 목표라고 하셨다. Django 템플릿으로 회원가입 폼을 만들어서 회원 정보를 DB로 보내주면 되는 것 같다.

<br>
<br>

## 해야할 일

![image](https://user-images.githubusercontent.com/84084372/197143194-5195fd16-b988-43d5-8207-cff3e6bcc7d5.png)

해당 사이트는 다음과 같이 교육과정이 존재한다. 따라서 사용자가 계정을 생성할 수 있는 경로는 2가지가 있다고 한다. 첫 번쨰는 관리자가 직접 사용자를 추가하는 방법이고(관리자>사용자), 또다른 방법은 각 과정별로 사용자를 추가하는 방법이다(사내교육>사용자). 원칙대로라면, 관리자가 최상위 테이블이 되어서 (관리자>사용자) 테이블이 (사내교육>사용자) 테이블을 포함하는 형태가 바람직해 보인다. 하지만, 현재 DB 구조는 그렇지 않을수도 있다고 하셨다. 따라서 해당 내용은 직접 확인해봐야할 것 같다.

<br>
<br>

## DB 접속

DB 툴은 DataGrip을 사용하였다. DB와 연결하기 위해 다음과 같이 DB 정보를 입력해야 한다. `docker-compose.dev.yml` 파일에있는 DB 포트 정보를 입력하여 접속하려고 했는데, 해당 포트는 docker에서만 사용하기 떄문에 접속이 안되고, localhost로 접속해야 한다고 하셨다.

![image](https://user-images.githubusercontent.com/84084372/197144751-40eb6469-c87a-494f-aa3b-66718f52e52c.png)

<br>
<br>

<div align='center'>
  <a href="https://on-jungwoan.github.io/internship/Itern-TIL-10/#1021">
    본문으로 돌아가기
  </a>
</div>
