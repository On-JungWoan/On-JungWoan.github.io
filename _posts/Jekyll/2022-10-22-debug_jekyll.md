---
title:  "[Jekyll]Jekyll 로컬에서 디버그 하는 법"
excerpt: "개발 블로그 로컬에서 디버깅"

categories:
  - Blog
tags:
  - [Github, Githubio, jekyll]


published: true

toc: true
toc_sticky: true
 
date: 2022-10-22
last_modified_at: 2022-10-22
til: 'true'
---

## 1. Git Clone

프로젝트 파일을 원하는 로컬 환경에 clone 해준다.

```
>>> git clone repo경로
```

<br>
<br>

## 2. Edit Gemfile

Gemfile을 수정해준다. 나는 minimal-mistakes를 사용하고 있어서 공식 문서를 참고하였다. 

- **Link** : [Remote theme methodPermalink](https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/#remote-theme-method)


```ruby
source "https://rubygems.org"

gem "github-pages", group: :jekyll_plugins
gem "jekyll-include-cache", group: :jekyll_plugins

gem "webrick", "~> 1.7"
```

<br>
<br>

## 3. Ruby 설치

아래 링크에서 Ruby를 설치해준다. 본인 os 환경에 맞춰서 설치해주면 된다.

- **Link** : [https://rubyinstaller.org/downloads/](https://rubyinstaller.org/downloads/)

<br>
<br>

## 4. 로컬서버 구동

Ruby 터미널에 아래 명령을 순차적으로 입력해준다.

```
## 인코딩 설정
>>> chcp 65001

## 라이브러리 설치
>>> gem install bundler jekyll minima jekyll-feed tzinfo-data rdiscount

## 서버 구동
>>> bundle exec jekyll serve
```

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/197326320-fb20ad92-6670-490b-94f9-3dde6ae31b98.png"></p>

<br>

localhost 4000번 포트로 접속하면 로컬에서 성공적으로 jekyll 디버깅을 할 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/197326461-13682eb7-d9a9-4002-9681-19300e2264db.png"></p>

<br>
<br>

## 5.에러유형
### 5-1) "No GitHub API authentication" error

```

GitHub Metadata: No GitHub API authentication could be found. Some fields may be missing or have incorrect data.

```

위와 같은 에러가 뜬다면 `config.yml`에 다음 내용을 추가해준다.

```yaml
github: [metadata]
```

<br>

### 5-2) cannot load such file -- webrick

```

C:/Ruby30-x64/lib/ruby/gems/3.0.0/gems/jekyll-4.2.0/lib/jekyll/commands/serve/servlet.rb:3:in `require': cannot load such file -- webrick (LoadError)

```

ruby 터미널에 다음과 같이 입력하면 된다.

```
>>> bundle add webrick
```

<br>

### 5-3) Could not find gem 'wdm (>= 0.1.0) x86-mingw32' in any of the gem sources listed in your Gemfile

대부분 linux 환경에서 bundle install시 발생하는데, 이는 권한이 없어서 발생하는 이슈이기 때문에 super user 권한으로 접근하면 해결된다.

```
>>> sudo bundle install
```

<br>

### 5-4) 기타 대부분의 오류

 `bundle update`를 해주면 대부분 해결된다.

```
>>> bundle update
```

<br>
<br>

- **Reference**

    [https://github.com/github/pages-gem/issues/399](https://github.com/github/pages-gem/issues/399)

    [이해 못 하고 쓰고보는 Jekyll](https://daki404.github.io/resource/2020/10/11/Resource-idea-Jekyll/)

    [Remote theme methodPermalink](https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/#remote-theme-method)

    [jekyll 오류 해결 (부제: github 블로그 만들기)](https://velog.io/@minji-o-j/jekyll-%EC%98%A4%EB%A5%98-%ED%95%B4%EA%B2%B0)