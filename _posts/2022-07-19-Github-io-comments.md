---
title:  "[Github_Io]utterances 댓글 추가 안될 때 해결법"
excerpt: "utterances를 사용한 댓글 기능 추가"

categories:
  - Blog
tags:
  - [jekyll, Github, Githubio]

toc: true
toc_sticky: true
 
date: 2022-07-19
last_modified_at: 2022-07-19
---

<br>

개발 블로그에 댓글 기능을 추가해보려고 한다.<br>
utterances를 사용하여 댓글 기능을 추가하려고 하는데, 인터넷에 나와있는 방법으로는 제대로 작동을 안해서 내가 삽질해서 얻은 노하우를 공유하고자 한다.<br>
해당 내용은 minimal-mistakes 사용자 기준으로 작성되었습니다.

<br>

## 0. utterances 설치
![image](https://user-images.githubusercontent.com/84084372/179539589-1b54e5a6-3441-4535-bf95-55389139ff76.png)

<br>

댓글 플랫폼 중 **utterances**가 가볍고 괜찮다고 하길래 해당 댓글 플랫폼을 사용하기로 했다. 
댓글을 달기 위해서는 깃허브 계정이 필요하다고 한다. 
방문자분들은 개인 github 계정을 가지고 계실 것으로 예상되기 때문에 무리없이 사용해도 될 것 같다. 
댓글에 마크다운도 사용 가능하다고 한다. 
아래 링크를 클릭해서 먼저 utterances를 설치한다. <br>
> [utterances설치](github.com/apps/utterances)

<br>

![image](https://user-images.githubusercontent.com/84084372/180615155-36699e51-ae94-4980-87ba-50dd1febf33d.png)

<br>

## 1. 댓글 이슈가 올라올 Repository 생성 <br>
나는 그냥 github.io Repository를 사용하였다. 
댓글이 달리면 해당 Repository에 issue로 올라온다고 한다. 
혹시 댓글 전용 Repository를 만들고 싶으면, 미리 생성하도록 하자. <br>
> [Repository링크](https://github.com/On-JungWoan/On-JungWoan.github.io/)

<br>

## 2. config.yml 수정 <br>
minimal-mistakes를 사용한다면, install에서 설정한 값대로  `config.yml` 파일을 수정해주면 된다.
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

<span style='background-color:#ffdce0'><strong>중요!</strong></span>

<span style='background-color:#ffdce0'>
  <strong>
    만약 defaults의 comments가 true로 되어있지 않다면, 꼭 true로 바꾸어준다. 
    그렇지 않으면 댓글이 표시되지 않으니 꼭 true로 바꾸어준다.
  </strong>
</span>

![image](https://user-images.githubusercontent.com/84084372/180615354-abf5ffd1-75c5-461b-953f-da2b2a1690bb.png)

<br>

## 3. 그래도 안 될 경우
위에서 설정한 세팅대로 `_includes/comments-providers/utterances.html`을 수정한다.

```html
  <script>
    'use strict';

    (function() {
      var commentContainer = document.querySelector('#utterances-comments');

      if (!commentContainer) {
        return;
      }

      var script = document.createElement('script');
      script.setAttribute('src', 'https://utteranc.es/client.js');
      script.setAttribute('repo', 'On-JungWoan/On-JungWoan.github.io'); # 본인 Repository를 정확하게 적어준다.
      script.setAttribute('issue-term', 'pathname');
      script.setAttribute('theme', 'github-light');
      script.setAttribute('label', 'blog-comment')
      script.setAttribute('crossorigin', 'anonymous');

      commentContainer.appendChild(script);
    })();
  </script>
```
