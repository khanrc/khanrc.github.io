---
layout: post
title: "RESTful API"
tags: ['Web']
date: 2014-07-27 13:08:00
---
**API**(Application Programming Interface)

API는 해당 어플리케이션에 대해 자세히 알지 못해도 API만 알고 있으면 어플리케이션의 기능을 활용할 수 있게 만들어 놓은 인터페이스다.

  


**[REST](http://ko.wikipedia.org/wiki/REST)**[(Representational State Transfer)](http://ko.wikipedia.org/wiki/REST)

World Wide Web과 같은 분산 하이퍼미디어[각주:1] 시스템을 위한 소프트웨어 아키텍쳐의 한 형식. REST원리를 따르는 시스템을 RESTful 하다고 한다.

엄격한 의미로 REST는 네트워크 아키텍처 원리의 모음이다. 여기서 네트워크 아키텍처 원리란 리소스를 정의하고 리소스에 대한 주소를 지정하는 방법에 대한 개괄을 말한다. 간단한 의미로는, 도메인 지향 데이터를 HTTP위에서 SOAP이나 쿠키를 통한 세션 트랙킹 같은 부가적인 전송 레이어 없이, 전송하기 위한 아주 간단한 인터페이스를 말한다. 이 두 가지의 의미는 당연히 겹치는 부분과 충돌되는 부분이 있다.

  


**RESTful API**

RESTful한 API, 즉 잘 디자인된 API는 설명을 간결하게 하고, 이해를 돕고, 여러 문제 상황(버전, 언어/포맷 선택 등)을 잘 해결할 수 있게 해준다. 

중심 규칙

  * URI는 정보의 자원을 표현해야 한다.
  * 자원에 대한 행위는 HTTP Method(GET, POST, PUT, DELETE 등)으로 표현한다.

GET: Select

POST: Insert

PUT: Update

DELETE: Delete

GET은 리소스를 가져오고, POST는 리소스를 생성하며, PUT은 리소스를 업데이트하고, DELETE는 리소스를 지운다.

  


즉, API의 스탠다드한 규격을 제시하며 이를 통해 사용의 이해와 다양한 문제해결을 도울 수 있다. 

이게 핫하게 떠오른 지 최소 3년 이상 된 거 같은데 특별히 중요한 지는 의문이다. 특히 린스타트업이 떠오르는 지금 스타트업에서 이런거에 목매일 필요는 없어 보인다.

그렇다 해도 개념은 잘 알아두자.

  


  


**마무리**

알수록 복잡하다. 지금은 이정도만 알아 두고, 나중에 필요하면 더 공부하기로 하자.

  


**참조**

http://bcho.tistory.com/321

http://spoqa.github.io/2012/02/27/rest-introduction.html?fb_action_ids=10204245754244709&amp;fb_action_types=og.comments

https://speakerdeck.com/leewin12/rest-api-seolgye

  1. 텍스트, 이미지, 동영상 등 미디어들이 서로 연결되어 이루어짐 [본문으로]


[Tistory 원문보기](http://khanrc.tistory.com/4)
