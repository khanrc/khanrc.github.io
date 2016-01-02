---
layout: post
title: "REST API 서버 제작기 (3) - API 변화"
tags: ['Web']
date: 2014-10-11 23:37:00
---
# REST API 서버 제작기 (3) - API 변화

여느 API들이 다들 그렇듯, 우리도 API를 정하는 과정에서 많은 변화가 있었다. 내부적인 시스템이 완성된게 아니기 때문에 앞으로도 계속 변화가 있을 수 있다는 사실을 글을 다 쓸 때쯤에 새삼 깨달았으나, 일단 이쯤에서 정리해 둔다.

## APIs

### 0\. 유저 토큰

**METHOD /API  
-H user_token: mytoken**

클라이언트는 서버에 리퀘스트를 보낼 때 항상 헤더에 유저 토큰을 넣어서 보낸다. SESSION ID 의 역할을 한다.

### 1\. 메인 카드 리스트

**GET /news/category/:category**

page를 없앴다. 본 서비스는 뉴스 추천 서비스이기 때문에, 시시각각 새로운 데이터가 업데이트되며, 유저의 리액션에 따라 컨텐츠의 순위가 바뀐다. 즉, 각 페이지별로 고정된 데이터가 아니다. 따라서 페이지별로 불러오는 것은 의미가 없으며 각 카테고리별로 서버에 요청을 하면 클라이언트가 받아보지 못한 새로운 데이터를 돌려준다.

이를 위해 서버는 클라이언트가 어떤 아티클들을 보았는지를 체크한다. 최근의 아티클들에 대해서만 체크하면 되므로 오버로드는 그리 크지 않을 것으로 본다.

### 2\. 뉴스 상세 페이지

**GET /news/articles/:article_id**

상세 페이지 정보를 불러오는 API는 초기에 생각했던 형태 그대로 간다.

### 3\. 유저 프로파일링

**POST /users/log**

원래 PUT이었지만 POST로 변경되었다. PUT/POST의 구분은 idempotent1 / not idempotent로 한다. put은 여러번 하더라도 같은 값을 유지해야 하고, post는 여러번 하면 계속 추가된다. 간단한 예를 들면 x=1은 put, x+=1은 post라 할 수 있다. 

유저 프로파일링은 한 유저의 프로파일을 계속 업데이트 한다는 관점에서는 put이지만, 유저의 리액션 정보를 데이터베이스에 계속 쌓는 개념이므로 post가 올바르다.

### 4\. 유저 등록

**POST /users**

신규 유저 등록은 당연히 POST. 앱을 처음 실행하면 이 메소드를 사용하여 유저 토큰을 넘겨 서버에서 신규 유저를 등록할 수 있도록 한다.

## State codes

API를 정의할 때 상태 코드도 같이 정의해 주어야 한다. 상태 코드는 크게 통신 상태 코드와 시스템 내부 상태 코드로 나눌 수 있다. 경계가 조금 애매하다는 생각이 드는데, 에러 코드 또한 추후 한번 더 검토하기로 한다.

### 통신 상태 코드

리퀘스트 형식이 잘못된 경우. 유저 토큰이 잘못되었거나 리퀘스트 바디가 이상하거나.  
이 경우엔 http error code를 이용한다.

### 시스템 에러

리퀘스트는 제대로 들어왔으나, 요청이 이상한 경우.

## API Server 개발

내부 시스템이 완성된 건 아니지만 일단 테스트용 api서버를 deploy해 놓기로 하였다.  
몇가지 추가된 부분을 정리했다.

### 유저 토큰 검사

먼저 유저 토큰을 검사하는 부분이 필요하다.
    
    
    from flask import request
    
    @app.before_request
    def before_filter():
        user_token = request.headers.get('user_token')
    

모든 api에 대해서 유저 토큰을 검사하므로 before_filter로 묶어 놓는다.  
유저 토큰은 헤더에 담기로 했고 헤더를 조회하기 위해 request모듈 import가 필요하다.

### Response
    
    
    from flask import Flask, Response
    
    @app.route('/news/articles/<int:article_id>', methods=['GET'])
    def article(article_id):
        ret = TEST JSON DATA...
    
        return Response(response=json.dumps(ret), 
                        mimetype='application/json', 
                        status=200)
    

테스트데이터를 넣어 두었다.  
리스폰스의 http 상태 코드와 mimetype을 조작하기 위해 Response모듈을 임포트했다.  
추후에는 db조회를 통해 데이터를 꺼내 유저에게 전송하는 부분이 들어갈 것이다.

* * *

  1. 멱등성. 연산을 여러번 하더라도 결과가 달라지지 않는 것. ↩


[Tistory 원문보기](http://khanrc.tistory.com/54)
