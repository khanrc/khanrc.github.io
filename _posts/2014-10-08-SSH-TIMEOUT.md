---
layout: post
title: "SSH TIMEOUT"
tags: ['Server(Back-end)']
date: 2014-10-08 23:41:00
---
# SSH TIMEOUT

서버 개발을 하다보면 ssh 타임아웃 설정은 당연히 하게 된다.

## first try

[khanrc: Ubuntu에 개발환경 셋팅하기 (2)](http://khanrc.tistory.com/entry/Ubuntu%EC%97%90-%EA%B0%9C%EB%B0%9C%ED%99%98%EA%B2%BD-%EC%85%8B%ED%8C%85%ED%95%98%EA%B8%B0-2)  
여기서 ssh timeout 설정을 했었다.
    
    
    $ cd /etc/ssh/
    $ sudo vim sshd_config
    
    ClientAliveInterval 600
    ClientAliveCountMax 3
    

그런데 왜인지 소마에서 사용하는 ucloud 서버에서는 이게 먹히지 않는다.

## second try

<http://abipictures.tistory.com/m/post/918>  
이 블로그의 1 - (2) **SSH 접속 타임아웃 값 조정** 을 참고하였다.
    
    
    $ sudo vim /etc/bash.bashrc
    
    ...
    # readonly export TMOUT=300
    

제일 아래의 readonly export TMOUT=300 을 주석처리 해주면 된다.

성공!

## why?

왜 첫번째 시도는 안 될까? 구글링을 해서 나오는 대부분의 자료는 `first try` 임에도 불구하고. 우선순위 문제인지, 아니면 우분투 버전 문제인지… 나중에 알아보기로 한다.


[Tistory 원문보기](http://khanrc.tistory.com/51)
