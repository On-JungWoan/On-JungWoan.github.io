---
title: "2022 하반기 ICT 인턴십 개요"
layout: archive
permalink: categories/Internship
author_profile: true
sidebar_main: true
---

<div class="list__item">
  <article class="archive__item" itemscope itemtype="https://schema.org/CreativeWork">
    <h2 class="archive__item-title no_toc" itemprop="headline">
        <a href="/til/TIL-12/" rel="permalink">[ICT 인턴십]2022년 12월 TIL</a>
    </h2>
    <!--{% include page__meta.html type=include.type %}-->
    <p class="page__meta"><i class="far fa-fw fa-calendar-alt" aria-hidden="true"></i> {{ "12 30 2022" | date: "%B %d %Y" }}</p>
    <p class="archive__item-excerpt" itemprop="description">{{ "회사에서 배운 내용들 정리" | markdownify | strip_html | truncate: 160 }}</p>
  </article>
</div>

<div class="list__item">
  <article class="archive__item" itemscope itemtype="https://schema.org/CreativeWork">
    <h2 class="archive__item-title no_toc" itemprop="headline">
        <a href="/til/TIL-11/" rel="permalink">[ICT 인턴십]2022년 11월 TIL</a>
    </h2>
    <!--{% include page__meta.html type=include.type %}-->
    <p class="page__meta"><i class="far fa-fw fa-calendar-alt" aria-hidden="true"></i> {{ "11 30 2022" | date: "%B %d %Y" }}</p>
    <p class="archive__item-excerpt" itemprop="description">{{ "회사에서 배운 내용들 정리" | markdownify | strip_html | truncate: 160 }}</p>
  </article>
</div>

<div class="list__item">
  <article class="archive__item" itemscope itemtype="https://schema.org/CreativeWork">
    <h2 class="archive__item-title no_toc" itemprop="headline">
        <a href="/til/TIL-10/" rel="permalink">[ICT 인턴십]2022년 10월 TIL</a>
    </h2>
    <!--{% include page__meta.html type=include.type %}-->
    <p class="page__meta"><i class="far fa-fw fa-calendar-alt" aria-hidden="true"></i> {{ "10 30 2022" | date: "%B %d %Y" }}</p>
    <p class="archive__item-excerpt" itemprop="description">{{ "회사에서 배운 내용들 정리" | markdownify | strip_html | truncate: 160 }}</p>
  </article>
</div>

<div class="list__item">
  <article class="archive__item" itemscope itemtype="https://schema.org/CreativeWork">
    <h2 class="archive__item-title no_toc" itemprop="headline">
        <a href="/til/TIL-09/" rel="permalink">[ICT 인턴십]2022년 09월 TIL</a>
    </h2>
    <!--{% include page__meta.html type=include.type %}-->
    <p class="page__meta"><i class="far fa-fw fa-calendar-alt" aria-hidden="true"></i> {{ "9 30 2022" | date: "%B %d %Y" }}</p>
    <p class="archive__item-excerpt" itemprop="description">{{ "회사에서 배운 내용들 정리" | markdownify | strip_html | truncate: 160 }}</p>
  </article>
</div>


{% assign posts = site.categories.Internship %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}