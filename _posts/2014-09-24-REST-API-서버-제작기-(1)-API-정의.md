---
layout: post
title: "REST API 서버 제작기 (1) - API 정의"
tags: ['Server(Back-end)']
date: 2014-09-24 23:45:00
---
# REST API 서버 제작기 (1) - API 정의

사실 서비스에서 필요한 API가 많지 않아서 고민한 경우의 수도 적다. 그래도 몇 가지 고민이 있었다.

적용한 프로젝트는 현재 소프트웨어 마에스트로에서 진행중인 뉴스 추천 서비스다. 본 서비스는 메인 페이지(또는 락스크린)에서 뉴스들을 보여주고, 뉴스를 터치하면 상세 페이지에서 전문을 읽을 수 있으며, 메뉴에서 카테고리를 고를 수 있다.

## 필요한 API들은?

그게 REST든 아니든, API 정의의 시작은 필요한 API의 나열 부터다.

  * 메인 카드 리스트  
메인 뉴스 리스트. 카테고리에 따라 나오는 뉴스들의 종류가 달라지며 페이지 기준으로 뉴스 아티클 리스트들을 전달받는다. 페이지당 10개의 아티클.

  * 뉴스 상세 페이지  
뉴스의 상세 정보

  * 유저 프로파일링  
유저가 어떤 아티클을 봤는지, 어떤 아티클을 오래 보고 있었는지, 어떤 카테고리를 좋아하는지 등등 유저 프로파일링 정보를 서버로 보내야 한다.

  * 유저 등록  
유저 등록에는 크게 두 가지가 있다. 

    1. 회원 등록 : UX 편의성을 최대화하기 위해, 회원 가입은 따로 하지 않는다. 디바이스 아이디 기반으로 유저를 구분한다.
    2. 디바이스 토큰 : Push를 보내려면 디바이스 토큰이 있어야 한다  
-&gt; 이 두가지를 합칠 수 있을 것 같다.
  * Global Setting  
기본적으로 버전정보가 필요할 것이다. 그 외에도 따로 필요한 것은?

## REST API

  * 메인 카드 리스트
    * GET /news/category/:category/page/:page
  * 뉴스 상세 페이지
    * GET /news/articles/:article_id
    * GET /news/category/:category/page/:page/articles/:article_id
  * 유저 프로파일링
    * PUT /users/:user_id/profile
  * 유저 등록
    * 이건 따로 API를 제공하는게 아니라 매 요청마다 클라이언트가 서버에게 키값을 넘기는 것으로 한다.
    * POST /users 또는 /users/:user_id/device_token (유저 구분자와 push를 위한 디바이스 토큰을 통합하느냐 나누느냐에 따라 달라짐)
  * 기타
    * Global setting
      * 버전정보
      * GET /global_setting

## 의문점

  * 사용하지 않을 요청에 대한 처리
    * POST, PUT, DELETE 라던가
    * 또는 articles 같은 요청
  * 뉴스 상세 페이지를 가져오는 요청은 뭐가 맞는가?
    * 사용하기 간편하고, 아티클아이디만 알면 가져올 수 있는 전자
    * 구조적으로 옳다고 할 수 있는 후자
  * news는 빼는게 맞는가?
    * 빼면 뭘 가져오는지 URI만 봐서는 알 수가 없지 않나?

## 참고

[Restful API 잘 정의하기](http://swguru.kr/50)  
[REST API 설계와 구현 (A 부터 I 까지)](http://seminar.eventservice.co.kr/JCO_1/images/track4-1.pdf)


[Tistory 원문보기](http://khanrc.tistory.com/41)
