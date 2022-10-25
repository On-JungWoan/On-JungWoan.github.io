---
title:  "Git 대용량 파일 push 방법 총정리"
excerpt: "과정 및 관련 에러 정리"

categories:
  - github
tags:
  - [git]

published: true

toc: true
toc_sticky: true
 
date: 2022-10-25
last_modified_at: 2022-10-25
---

## 1. 문제상황

100MB를 넘는 파일을 push하려고 하면, 다음과 같은 에러가 발생한다.

```
remote: error: this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
```

이를 해결하기 위해서는 유료 github 계정을 구입해야하는데, 그렇지 않고 해결할 수 있는 방법을 소개한다.

<br>
<br>

## 2. 해결방법
### 2-1) 대용량 파일을 커밋하기 전인 경우

`git lfs`를 사용하여 대용량 파일을 추적한다.

- **git lfs?**

  100MB가 넘는 대용량 파일이 있을 떄 해당 파일을 작게 조각내어 commit할 수 있게 해주는 tool이다.

git bash에 다음 코드를 순차적으로 입력하여 대용량 파일을 lfs가 추적할 수 있게 한다.

```
# lfs 설치
>>> git lfs install

# 대용량 파일 add 기록 unstaging -> git 추적 중지
>>> git rm -r --cached "대용량 파일의 경로"

# 대용량 파일 추적
>>> git lfs track "대용량 파일의 경로"
```

그 후 `add / commit / push`를 하면 정상적으로 push가 된다.

```
>>> git add *
>>> git commit -m "msg"
>>> git push origin branch이름
```

<br>

### 2-2) 이미 대용량 파일을 커밋한 경우

이 경우는 `bfg`를 설치하여 기존 커밋 내역에서 대용량 파일에 대한 commit 내역을 삭제해야한다. 우선, 아래 링크에 들어가서 bfg를 설치해준다.

- **[bfg 설치]** : [https://rtyley.github.io/bfg-repo-cleaner/](https://rtyley.github.io/bfg-repo-cleaner/)

또한, bfg는 java 기반이므로, JRE를 설치해주어야 한다. JRE는 아래 링크에서 설치하면 된다.

- **[JRE 설치]** : [https://www.java.com/ko/download/ie_manual.jsp?locale=ko](https://www.java.com/ko/download/ie_manual.jsp?locale=ko)


위에서 다운 받아준 bfg를 프로젝트 경로에 넣어준다. 그리고 git bash 또는 터미널에 다음 명령을 입력한다. 여기서 bfg-1.14.0.jar는 위에서 다운받은 bfg의 이름을 넣어주면 된다.

```
# 100M 이상의 대용량 파일에 대한 기록 삭제
>>> java -jar bfg-1.14.0.jar --strip-blobs-bigger-than 100M
```

이후 push하면 해결된다.

```
git push origin branch이름
```

<br>

### 2-3) bfg 에러 및 해결방법

#### **Unable to access jarfile bfg-1.14.0.jar**

- `bfg-1.14.0.jar`의 정확한 경로를 적어주면 된다.

  ```
  >>> java -jar c:/users/user/desktop/project/bfg-1.14.0.jar --strip-blobs-bigger-than 100M
  ```

#### **Warning : no large blobs matching criteria found in packfiles**

- 100MB 이상의 파일을 찾을 수 없다고 나오면, 다음 명령어를 수행하고 다시 2-2의 명령을 수행하면 된다.

    ```
    >>> git repack && git gc
    ```

<br>
<br>

## 3. 성공!

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/197827271-81d3fb32-f2ee-4c14-a2fa-763cd7c2f6a3.png"></p>


<br>
<br>

<div align='center'>
  <a href="https://on-jungwoan.github.io/internship/Itern-TIL-10/#1025">
    본문으로 돌아가기
  </a>
</div>  

<br>
<br>

- **Reference**

  [Github에 100MB 이상의 파일을 올리는 방법](https://medium.com/@stargt/github%EC%97%90-100mb-%EC%9D%B4%EC%83%81%EC%9D%98-%ED%8C%8C%EC%9D%BC%EC%9D%84-%EC%98%AC%EB%A6%AC%EB%8A%94-%EB%B0%A9%EB%B2%95-9d9e6e3b94ef)