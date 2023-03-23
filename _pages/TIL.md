---
title: "TIL 내용 모음"
layout: archive
permalink: /TIL/
author_profile: true
sidebar_main: true
---

<div class="list__item">
  <article class="archive__item" itemscope itemtype="https://schema.org/CreativeWork">
    <h2 class="archive__item-title no_toc" itemprop="headline">
        <a href="https://docs.google.com/document/d/1fK2cAnHG0R7F2o_4SfEdJE6wLyNQb-a5HzPtKU9r39s/edit" rel="permalink">UVLL Daily research report</a>
    </h2>
    <!--{% include page__meta.html type=include.type %}-->
    <p class="page__meta"><i class="far fa-fw fa-calendar-alt" aria-hidden="true"></i> {{ "03 24 2023" | date: "%B %d %Y" }}</p>
    <p class="archive__item-excerpt" itemprop="description">{{ "UNIST UVLL 연구실 daily report" | markdownify | strip_html | truncate: 160 }}</p>
  </article>
</div>

{% assign posts = site.categories.TIL %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}
