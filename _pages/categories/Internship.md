---
title: "2022 하반기 ICT 인턴십 개요"
layout: archive
permalink: categories/Internship
author_profile: true
sidebar_main: true
---

<!-- <div class="list__item">
  <article class="archive__item" itemscope itemtype="https://schema.org/CreativeWork">
    <h2 class="archive__item-title no_toc" itemprop="headline">
        <a href="/til/TIL-12/" rel="permalink">[ICT 인턴십]2022년 12월 TIL</a>
    </h2>
    <p class="page__meta"><i class="far fa-fw fa-calendar-alt" aria-hidden="true"></i> {{ "12 30 2022" | date: "%B %d %Y" }}</p>
    <p class="archive__item-excerpt" itemprop="description">{{ "회사에서 배운 내용들 정리" | markdownify | strip_html | truncate: 160 }}</p>
  </article>
</div> -->

<!-- 나머지 9월~11월 내용은 전부 지웠음.
만약 복구하고 싶다면 주석처리 된 부분 풀어서 month 관련된 부분만 수정하면 될듯 -->

{% assign posts = site.categories.Internship %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}