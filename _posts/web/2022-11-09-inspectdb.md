---
title:  "[Django] 이미 존재하는 DB 연동 by inspectdb"
excerpt: "inspectdb 소개"

categories:
  - web
tags:
  - [Django, web, backend]

toc: true
toc_sticky: true
 
date: 2022-07-15
last_modified_at: 2022-07-15  
---

## 1. DB 불러오기

DB가 이미 존재하는 경우 `inspectdb`를 사용하여 기존 DB 정보를 불러올 수 있다. 터미널에 아래 명령을 입력하면 기존 DB에 대한 models.py가 최상위 경로 아래 old_models.py에 저장된다. 여기서 필요한 부분만 발췌하여 사용하거나, 전체를 복사하여 models.py에 저장해준다.

```
>>> python manage.py inspectdb > old_models.py
```

<br>

## 2. Models.py 수정

초기에는 managed=False로 되어 있기 때문에, 추후 모델 변경시 migrate 하기가 곤란하다. 따라서 managed=True로 수정해준다. 또한, charField의 max_length가 가변으로 되어 있을 경우, max_length=-1로 불러와져서 `"max_length" attribute that is a positive integer` 에러가 발생한다. 따라서 해당 부분들은 Django에 맞게 수정해준다.

```python
from django.db import models

class Users(models.Model):
    id = models.BigAutoField(primary_key=True)
    pronouns = models.CharField(max_length=255, blank=True, null=True)

    (..중략..)

    class Meta:
        managed = True
        db_table = 'users'
        # unique_together = (('root_account_ids', 'id'),)
```

<br>

## 3. Fake migrate

이미 테이블이 존재하기 때문에 따로 migrate를 해줄 필요가 없으므로, fake migrate를 해준다. 

```
>>> python manage.py migrate --fake-initial
```

<br>

## 4. Migrate

이후에 모델 변경시에는 기존과 동일하게 makemigrations -> migrate 해주면 된다.

<br>

## 5. 정리

1. python manage.py inspectdb > 불러올 파일 경로

2. managed=True, max_length 수정

3. python manage.py migrate --fake-initial

4. 모델 변경 시 makemigrations -> migrate

<br>
<br>

<div align='center'>
  <a href="https://on-jungwoan.github.io/internship/Intern-TIL-11/#1109">
    본문으로 돌아가기
  </a>
</div>
