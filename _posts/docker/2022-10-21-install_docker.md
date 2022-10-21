---
title:  "Window Docker 설치방법"
excerpt: "windows for docker"

categories:
  - docker
tags:
  - [Docker]

---

## 1. 환경설정
### 1-1) 가상화 사용설정
작업관리자 -> 성능 -> CPU의 가상화가 사용으로 되어있는지 확인한다. 만약 사용으로 되어 있지 않으면, BIOS에서 사용함으로 설정해야한다.

![image](https://user-images.githubusercontent.com/84084372/197147783-69f00da2-4766-4a39-a2f9-bafd88c154b5.png)

<br>

### 1-2) Hyper-V 켜기

`window + s` -> `프로그램 추가/제거` -> `선택적 기능` -> `기타 Windows 기능` -> `Hyper-V 체크하기`

![image](https://user-images.githubusercontent.com/84084372/197148232-261e8ceb-5cfd-403b-822f-4dde9703aa95.png)
![image](https://user-images.githubusercontent.com/84084372/197148263-ef727ba4-ecbf-46df-a576-387df2c96338.png)
![image](https://user-images.githubusercontent.com/84084372/197148290-e634fdec-61b1-455c-80ec-b66704a465b8.png)

<br>
<br>

## 2. Docker 설치

### 2-1) Docker installer 설치

 - [https://hub.docker.com/editions/community/docker-ce-desktop-windows/](https://hub.docker.com/editions/community/docker-ce-desktop-windows/)

Docker Desktop for Windows를 클릭하여 installer 설치 후, install하면 된다.

![image](https://user-images.githubusercontent.com/84084372/197149147-08304f83-c94a-452c-8581-bed63533ee4c.png)

- **docker 실행 시 에러**

Docker 실행시에 다음과 같은 에러가 뜬다면 화면에 보이는 사이트에 접속해서 업데이트 패키지를 다운받아주면 된다.

![image](https://user-images.githubusercontent.com/84084372/197148579-a53edc96-dcf0-40c0-a1b0-00a043289a96.png)

![image](https://user-images.githubusercontent.com/84084372/197148633-f56da1d5-0938-4295-8ebe-4d3ce4d20abd.png)

<br>

### 2-2) 설치 완료

cmd에 `docker -v`를 입력했을 때, docker 버전이 표시되면 설치 완료이다.

![image](https://user-images.githubusercontent.com/84084372/197148698-3a4ffe68-5307-4750-96d4-94b28e3c1f31.png)

<br>
<br>

<div align='center'>
  <a href="https://on-jungwoan.github.io/internship/Itern-TIL-10/#1021">
    본문으로 돌아가기
  </a>
</div>
