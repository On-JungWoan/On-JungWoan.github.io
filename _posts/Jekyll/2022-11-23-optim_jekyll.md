---
title:  "[Jekyll]빌드 속도 최적화"
excerpt: "다양한 방법을 사용하여 Jekyll 빌드 시간 개선"

categories:
  - web-development
tags:
  - [Githubio, jekyll, html]


published: true

toc: true
toc_sticky: true
 
date: 2022-11-23
last_modified_at: 2022-11-23
til: 'true'
permalink: "/blog/optim_jekyll/"
legacy_category: Blog
---

{% raw %}

## 1. 최적화 방법

Hugo 대비 Jekyll의 가장 큰 단점은 바로 빌드 시간이 매우 느리다는 것이다. 포스팅의 개수가 늘어날수록 빌드 시간은 매우 느려져서, 혹자의 경우 빌드 시간이 10분을 초과하는 경우도 있다고 한다. 따라서 해당 포스팅에서는 Jekyll 빌드 시간을 최적화하는 방법에 대해 소개한다.

### 1-1) liquid-c

liquid-c를 설치하면, liquid문을 C로 처리하면서 빌드 속도를 올려준다고 한다. gem install을 사용하여 쉘에서 직접 설치하거나, gemfile 작성 후 bundle install로 liquid-c를 설치해주면 끝이다.

- **1. 직접 설치**

```ruby
gem install liquid-c
```

- **2. gemfile 작성**

```ruby
gem "liquid-c"
```

```
bundle install
```

<br>

### 1-2) incremental 옵션 변경

기존 jekyll은 하나의 파일만 수정되어도, 전체 파일을 전부 빌드한다. 따라서 포스트 개수가 많아질수록 빌드 시간도 오래 걸린다. 필자의 경우, 포스팅 100개 기준 빌드 시간은 약 15초정도였다. 따라서 바뀐 부분만 빌드되게 하여 빌드 속도를 크게 향상시킬 수 있다. `_config.yml`의 `incremental`옵션을 true로 바꾸어주면 자동으로 적용된다.

<br>

### 1-3) limit post 설정

jekyll 빌드 시, `--limit_posts` 플래그를 사용하면, post 개수에 제한을 걸어 빌드할 수 있다. 가령 터미널에 다음과 같이 입력했다고 가정하면, 작성일자 기준 최신 3개의 포스팅만 볼 수 있다. 포스팅 디버깅할 떄 아주 유용하게 사용할 수 있을 것 같다.

```ruby
bundle exec jekyll serve --limit_posts 3
```

<br>
<br>

## 2. 속도 차이

생각보다 속도 차이가 엄청나서 깜짝 놀랐다. 빨라져봤자 3~4초정도일 줄 알았는데, 거의 15배 가량의 눈부신 성능 향상을 보여주었다.

<div align="center"><strong>[최적화 전]</strong></div>
<p align="center"><img src="https://user-images.githubusercontent.com/84084372/203552958-6d710312-cf2b-487e-951f-10cb1ef022ad.png" style="border: 1px solid black"></p>

<div align="center"><strong>[최적화 후]</strong></div>
<p align="center"><img src="https://user-images.githubusercontent.com/84084372/203555535-3e3e189b-ccfb-42c8-ad58-7285ddb7a2d8.png" style="border: 1px solid black"></p>

{% endraw %}
