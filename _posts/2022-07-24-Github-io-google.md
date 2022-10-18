---
title:  "[Github_Io]개발 블로그 Google에 노출시키기"
excerpt: "개발 블로그 google 노출"

categories:
  - Blog
tags:
  - [jekyll, Github, Githubio]

published: true

toc: true
toc_sticky: true
 
date: 2022-07-24
last_modified_at: 2022-07-24
---


<br>

개발 블로그에 작성한 글들을 다양한 사람들과 공유하기 위해서 Google에 노출시키고자 한다.<br>
다음 일련의 과정은 해당 내용을 담고있다.

<br>

## 1. Google Search Console
google에 해당 기술 블로그를 노출시키기 위해서 `google search console`을 사용합니다. 
`google search console`은 google에서 검색시, 나의 개발 블로그가 보여질 수 있도록 등록하는 google 서비스입니다.

![image](https://user-images.githubusercontent.com/84084372/180615935-ddf44ac8-11cd-4a62-bdb0-e801985813e4.png)

<br>

화면의 시작하기 버튼을 눌러서 시작합니다.

<br>

![image](https://user-images.githubusercontent.com/84084372/180615984-71dec1c4-2115-47fc-b8fc-502d21bdf108.png)

<br>

URL 접두어에 본인의 개발 블로그 주소릅 입력합니다.

<br>

![image](https://user-images.githubusercontent.com/84084372/180616041-2a04d0d2-7ba4-4683-9511-d9959cd8320e.png)

<br>

그 후 다음과 같이 화면에 뜨는 html 파일을 다운로드 받아서, `config.yml`이 위치한 경로에 넣어줍니다.

<br>

## 2. HTML 파일 업로드
위에서 다운로드 받은 html 파일을 다음과 같이 root 디렉토리에 업로드합니다.

![image](https://user-images.githubusercontent.com/84084372/180616092-84cccfa1-f1b0-4a45-8f44-dff8c75e6923.png)

<br>

## 3. sitemap.xml 생성
`cofig.yml`이 위치한 경로에 다음과 같은 `sitemap.xml`을 생성하여줍니다.<br>
> 아래 스크립트에서, `<%`, `%>`의 `<`, `>`는 `{`, `}`로 바꾸어서 사용하여줍니다.

<br>

```xml
---
layout: null
---

<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd"
        xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <% for post in site.posts %>
    <url>
        <loc>{{ site.url }}{{ post.url }}</loc>
        <% if post.lastmod == null %>
        <lastmod>{{ post.date | date_to_xmlschema }}</lastmod>
        <% else %>
        <lastmod>{{ post.lastmod | date_to_xmlschema }}</lastmod>
        <% endif %>

        <% if post.sitemap.changefreq == null %>
        <changefreq>weekly</changefreq>
        <% else %>
        <changefreq>{{ post.sitemap.changefreq }}</changefreq>
        <% endif %>

        <% if post.sitemap.priority == null %>
        <priority>0.5</priority>
        <% else %>
        <priority>{{ post.sitemap.priority }}</priority>
        <% endif %>

    </url>
    <% endfor %>
</urlset>
```
<br>

## 4. robots.txt 생성
마찬가지로 `_config.yml`이 위치한 경로에 다음과 같은 `robots.txt`를 생성하여준다.

```
User-agent: *
Allow: /

Sitemap: https://on-jungwoan.github.io/sitemap.xml
```

<br>

## 5. sitemap.xml 등록
다음과 같이 최종적으로 sitemap.xml을 등록하여준다.

![image](https://user-images.githubusercontent.com/84084372/180616337-45c2b745-5a1a-4f5b-a639-6a3557244ba7.png)

<br>

sitemap.xml을 등록하고 제출버튼을 누르면, 아래 사진과 같이 상태에 `성공`이라고 뜬다.

<br>

![image](https://user-images.githubusercontent.com/84084372/180616362-2840df4a-90df-429c-a256-631e41b9fa28.png)

## 6. 결과 대기
구글 노출까지 적게는 3~5일에서 길게는 한 달정도 걸린다고 한다.<br>
여유를 가지고 천천히 기다려보도록 하자!!
