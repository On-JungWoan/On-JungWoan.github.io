---
title: "기타 잡다한 내용들"
layout: archive
permalink: categories/etc
author_profile: true
sidebar_main: true
---
{% assign filtering = '캄보디아' %}
{% assign posts = site.categories.etc %}

{% for post in posts %}
    {% if post.title != filtering %}
        {% include archive-single.html type=page.entries_layout %}
    {% endif %}
{% endfor %}
