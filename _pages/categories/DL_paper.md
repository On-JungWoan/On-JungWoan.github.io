---
layout: category
title: "Paper Review"
type: grid
permalink: categories/DL_paper
author_profile: true
sidebar_main: true
---

{% assign filtering = 'private' %}
{% assign posts = site.categories.DL_paper %}

{% for post in posts %}
    {% unless post.title contains filtering %}
        {% include archive-single.html type=page.entries_layout %}
    {% endunless %}
{% endfor %}