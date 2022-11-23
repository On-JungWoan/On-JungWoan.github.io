---
title:  "[Jekyll]포스트에 특정 문구 고정(초안)"
excerpt: "post에 특정한 문구를 고정시키는 법"

categories:
  - Blog
tags:
  - [Githubio, jekyll, html]


published: true

toc: true
toc_sticky: true
 
date: 2022-11-21
last_modified_at: 2022-11-21
til: 'true'
---

{% raw %}

우선, 우리가 게시물을 작성하면 `게시물 -> single.html -> default.html`의 순서로 layout이 상속되어 화면에 출력된다. 여기서, 게시물의 직접적인 요소가 담겨있는 layout은 single.html이다. 따라서 해당 템플릿의 내용을 적절히 수정해주면 포스팅 형식을 바꿀 수 있다.

## ㅁ

single.html의 중간의 {{content}}가 포스팅 내용이 들어가는 영역이다. 따라서 해당 부분 전후로 원하는 문구를 넣으면, 포스팅에 해당 문구가 고정되어 출력된다.

```html
        (중략)

        {{ content }}

        {% if page.til %}
          {% capture date %}{{page.date | remove: '-'}}{% endcapture %}
          {% capture month %}{{ date | slice: 4, 2 }}{% endcapture %}
          {% capture month_day %}{{ date | slice: 4, 4 }}{% endcapture %}        
          <br><br><div align='center'><a href="https://on-jungwoan.github.io/til/TIL-{{month}}/#{{month_day}}">본문으로 돌아가기</a></div>
        {% endif %}
```

{% endraw %}