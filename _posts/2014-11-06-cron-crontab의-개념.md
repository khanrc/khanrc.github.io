---
layout: post
title: "cron, crontab의 개념"
tags: ['Web']
date: 2014-11-06 12:46:00
---
# cron, crontab, anacron

크론의 사용법 자체는 구글링 하면 널려 있는데, 이 설정을 어디에 저장해야 되냐라던지 개념적으로 애매한 부분이 많아서 정리했다.

## what's different?

`cron`은 타이머 데몬이고, `crontab`은 작업 설정 파일을 말한다. `crontab`명령어는 이 작업 설정 파일에 접근한다.

## [cron](http://man7.org/linux/man-pages/man8/cron.8.html)

### NAME

crond - 스케쥴된 커맨드를 실행하는 데몬.

### DESCRIPTION

크론은 _/etc/rc.d/init.d_나 _/etc/init.d_로 실행할 수 있다. `systemd`를 사용가능하다면, _systemctl start crond.service_로 실행가능하다. _'&amp;'_ 파라메터는 필요없다.

크론은 _/var/spool/cron/_에서 크론탭 파일을 찾는다. _/etc/passwd_에 등록된 계정 이름으로 계정별 크론탭 파일이 설정되어 있다. 또한 크론은 _/etc/crontab_을 찾고 _/etc/cron.d/_ 디렉토리의 파일을 찾는다 - 이 둘은 다른 포맷으로 되어 있다([crontab(5)](http://man7.org/linux/man-pages/man5/crontab.5.html) 참고). 크론은 각 크론탭을 검사하여 'current minutes'에 실행되어야 하는지 확인한다.

크론이 크론탭 파일이 수정되었는지 어떻게 알까? 여기에는 두 가지 방법이 있다. 첫째는 크론탭 파일의 _modtime(modified time)_을 확인하는 것이다. 두번째는 _inotify support_를 사용한다. _inotify_는 모든 크론탭 파일을 체크하고, 수정이 발생하면 알려준다.

_modtime_ 옵션을 사용하면 크론은 모든 크론탭 파일을 매 분마다 검사하며 변경이 감지되면 리로드한다. 크론을 재시작할 필요는 없다. 

크론은 아래 파일과 디렉토리들을 검사한다:

  * _/etc/crontab_  
system crontab. 요새는 안 쓰고 _/etc/anacrontab_ 이라는 config파일을 사용한다고 하는데 현재 우리 서버인 우분투 14.04에는 그런거 없다.
  * _/etc/cron.d/_  
각 유저별 시스템 크론잡을 저장한다.
  * _/var/spool/cron_  
`crontab`커맨드에 의해 저장된 유저 크론테이블을 저장한다.

## crontab

`crontab` 은 리눅스 메뉴얼이 두 개다. 유저 커맨드인 [crontab(1)](http://man7.org/linux/man-pages/man1/crontab.1.html), 파일 포맷인 [crontab(5)](http://man7.org/linux/man-pages/man5/crontab.5.html).

`crontab`은 _crontable_의 줄임말이다(그런 거 같다). 크론탭 커맨드는 크론이 사용하는 크론테이블들을 관리한다. 각 유저는 자신만의 crontab 파일을 갖고 이것은 각각 _/var/spool/_ 에 저장된다.

_cron.allow_, _cron.deny_ 파일을 통해 크론을 유저마다 allow/disallow 할 수 있다. _cron.allow_ 파일이 존재한다면 크론을 사용할 수 있는 유저는 여기에 등록되어야 하고, _cron.deny_ 파일이 존재한다면 그 반대다. 둘 다 없다면 오직 super user, 즉 _root_ 만이 크론을 사용할 수 있다.

## [anacron](http://www.fis.unipr.it/pub/linux/redhat/9/en/doc/RH-DOCS/rhl-cg-ko-9/s1-autotasks-anacron.html)

`cron`과 유사하지만 시스템이 켜져있지 않아도 작동한다. default로 설치되어 있는 건 아닌 것 같고 따로 설치해 주어야 한다. `rpm -q anacron`로 설치되어 있는지 확인할 수 있다.


[Tistory 원문보기](http://khanrc.tistory.com/67)
