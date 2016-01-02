---
layout: post
title: "Docker & Vagrant"
tags: ['Web']
date: 2014-11-02 22:45:00
---
# Docker &amp; Vagrant

기존의 프로젝트를 새로운 서버에 올린다고 생각해보자. 서버를 파고, `add user`라던가 `port forwarding`등 기본 세팅을 한 후, 프로젝트 세팅을 해야 한다. 소마 프로젝트를 예로 들자면, `python`을 설치하고, `python`에서 사용하는 라이브러리들인 `flask`, `nltk`, `newspaper` 등등 수많은 라이브러리들을 설치한다. 뿐만 아니라 `mariadb`, `mongodb`, `redis`등 데이터베이스들도 설치하고, 파이썬에서 이 데이터베이스에 커넥트하기 위해 사용하는 라이브러리들도 또 설치해야 한다. 또한 메시지 큐로 사용하는 `rabbitmq`도 설치하고, 위 과정과 마찬가지로 또 파이썬용 라이브러리를 설치해야 한다. 여기까지만 해도 충분히 많은데 이게 끝이 아니다. 이걸 다 설치했으면 소스를 가져와야 하는데, 일단 `git`을 먼저 설치하고, `git clone`을 통해 프로젝트를 가져와야 한다. 뿐만 아니라 프로젝트를 실제로 디플로이 하기 위해서 웹서버인 `apache`나 `nginx`를 설치하고, 파이썬 어플리케이션 컨테이너인 `wsgi` 모듈도 설치해야 한다.

이정도 하고 나면 이제야 이 서버를 돌릴 수 있다. 당연한 얘기지만 이 과정에서 실수가 나오고 문제가 발생하기 마련이며, 미처 적지 못한 과정도 있을 것이다. 즉, 한마디로 엄청 머리아프다. 지금까지 얘기한 이 모든 과정을 바로 `배포`, 즉 `deploy`라 한다.

마치 `git`이 프로젝트 소스 전체를 컨트롤하듯, `deploy`의 과정 전체를 컨트롤 할 수 있는 도구가 바로 `docker`다.

## what is docker?

[조대협의 블로그: Docker 소개](http://bcho.tistory.com/805)  
[khanrc: 가상화 이야기](http://khanrc.tistory.com/entry/%EA%B0%80%EC%83%81%ED%99%94-%EC%9D%B4%EC%95%BC%EA%B8%B0)

![VM](http://cfile8.uf.tistory.com/image/213CAE3D52655494054CF8)

VMWare나, Parallel desktop과 같이 우리가 평소에 사용하는 `Virtual Machine`은 이러한 형태로 동작한다. 먼저 `Host OS`위에 `Hypervisor`가 깔린다. 하이퍼바이저는 단일 시스템을 분할 및 관리한다. 이 위에 `Virtual Machine`이 올라가는데, 버추얼 머신은 하드웨어를 통째로 가상화한다. 이 가상 하드웨어 위에 새로운 `Guest OS`가 올라가는 것이 바로 `Virtual Machine`의 구조다.

![Container](http://cfile5.uf.tistory.com/image/272F203F5265549F04419F)

`docker`는 이와 달리, 하드웨어를 가상화하지도 않으며 다른 OS를 통째로 올리지도 않는다. 도커는 리눅스 컨테이너로서 OS간의 차이를 저장한다. 즉 `Host OS`가 우분투고 컨테이너에 CentOS를 올렸다면, 이 두 OS의 `diff`만을 따로 패키징하는 것이다. 이것이 바로 `Guest OS`의 `Isolation`이다. 이 덕분에 기존의 가상화와는 달리 따로 하드웨어 가상화를 거치지 않아 매우 가벼우며, 컨테이너 내에서의, 즉 `Guest OS`의 퍼포먼스가 `Host OS`와 별반 차이가 나지 않는다는 점이 특장점이다.

## Vagrant

[조대협의 블로그: Vagrant를 이용한 개발환경 관리(간단한 VM관리)](http://bcho.tistory.com/806)  
[Slideshare: Vagrant로 서버와 동일한 개발환경 꾸미기](http://www.slideshare.net/kthcorp/h3-2012-vagrant)  
`vargant`는 **간소화된 VM 관리 서비스**다. 내가 사용하는 맥에, 서버와 동일한 환경을 구축하기 위해 우분투를 올린다고 생각해보자. VMWare 따위의 `Hypervisor`를 이미 설치해 두었다고 해도, 먼저 가상 하드웨어 머신을 생성해야 한다. 그리고 우분투 이미지를 가져와 그 위에 우분투를 설치해야 한다. 우분투가 설치되고 나면 기본 설정을 해주어야 한다. 이러한 노가다성 작업을 손쉽게 해결하기 위한 프로젝트가 바로 `vargant`다. 크게 보면 `docker`와 아이디어 면에서 비슷한 점이 많다.
    
    
    // VM의 이미지와 기본 설정을 가져온다
    $ vagrant box add precise32 http://files.vagrantup.com/precise32.box
    $ vagrant init precise32
    
    // VM을 설치하고 구동시킨다
    $ vagrant up
    
    // ssh를 통해 VM에 로그인한다. 둘 중 아무거나 써도 됨.
    $ vagrant ssh
    $ ssh localhost:2222
    

  * box  
기본 OS 이미지 + 기본 설정(CPU, 메모리, 네트워크 등)
  * vagrant file  
vagrant init를 하면 생성된다. 어떤 box를 사용하는지, box에서 설정된 기본 설정들이 어떠한지 등이 명시되어 있다.
  * vagrant up  
위에서 정의한 설정대로 VM을 생성한다.

좀 더 세부적으로 들어가면 provisioning 등 많은 듯 싶지만, 이쯤에서 정리한다.


[Tistory 원문보기](http://khanrc.tistory.com/64)
