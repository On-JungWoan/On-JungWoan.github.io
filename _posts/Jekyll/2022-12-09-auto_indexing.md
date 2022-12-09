---
title:  "[Google search console] 구글 서치 콘솔 색인 생성 자동화 하는 법"
excerpt: "How to Automate the Indexing Process on Google Search Console"

categories:
  - Blog
tags:
  - [Githubio, jekyll, html]


published: true

toc: true
toc_sticky: true
 
date: 2022-12-09
last_modified_at: 2022-12-09
til: 'true'
---

## Github Actions

```yaml

name: Python package

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build:

    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.8", "3.9", "3.10"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install flake8 pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

```