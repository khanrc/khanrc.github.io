---
layout: post
title: "WSGI, uwsgi"
tags: ['Server(Back-end)']
date: 2014-07-27 17:31:00
---
[**WSGI**(Web Server Gateway Interface)](http://ko.wikipedia.org/wiki/%EC%9B%B9_%EC%84%9C%EB%B2%84_%EA%B2%8C%EC%9D%B4%ED%8A%B8%EC%9B%A8%EC%9D%B4_%EC%9D%B8%ED%84%B0%ED%8E%98%EC%9D%B4%EC%8A%A4)

**웹서버와 웹어플리케이션이 어떤 방식으로 통신하는가에 관한 인터페이스**를 의미한다. 웹서버와 웹어플리케이션 간의 소통을 정의해 어플리케이션과 서버가 독립적으로 운영될 수 있게 돕는다. WSGI는 파이썬 표준인 PEP333, PEP3333에 의해 제안되었고, 이 이후에 여러 언어로 구현된 프로젝트가 생겨나기 시작했다.

기존의 웹앱은 웹서버를 선택하는데 있어서 제약이 있었다. 보통 CGI, FastCGI, mod_python 과 같은 커스텀API 중에 하나만 사용할 수 있도록 디자인 되었는데, WSGI는 그에 반하여 low-level로 만들어져서 웹서버와 웹 어플리케션, 프레임워크간의 벽을 허물었다. 즉, WSGI가 웹서버와 웹앱 간의 미들웨어로 기능함으로써 웹서버와 웹앱이 각각 독립적으로 구동할 수 있게 해준다.

WSGI 어플리케이션은 uWSGI라는 컨테이너에 담아 어플리케이션을 실행하게 되며, uWSGI가 각각의 웹서버와 소통하도록 설정하면 끝이다. Flask, django와 같은 프레임워크는 이미 WSGI 표준을 따르고 있기 때문에 바로 웹서버에 연결해 사용할 수 있다.

  


**uWSGI**

현재 가장 좋은 퍼포먼스를 보이고 널리 사용되는 WSGI컨테이너다.

  


* WSGI가 파이썬에 한정되어있는 개념인지 아닌지 잘 모르겠음

파이썬에서 나왔고, 아직 다른 언어로 널리 퍼지진 않은 것 같다.

  


**참조**

http://software-engineer.gatsbylee.com/python-wsgi-server%EC%99%80-%EA%B4%80%EB%A0%A8%EB%90%9C-%EA%B3%B5%EB%B6%80/

http://spoqa.github.io/2011/12/24/about-spoqa-server-stack.html


[Tistory 원문보기](http://khanrc.tistory.com/7)
