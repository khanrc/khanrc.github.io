---
layout: post
title: "추천 시스템(Recommendation System)"
tags: ['DataScience']
date: 2014-08-30 05:54:00
---
이번 소프트웨어 마에스트로 과정 1차 프로젝트에서 가장 중요한 건 추천 시스템이다. 추천 시스템은 본 과정 뿐만 아니라 수많은 스타트업들의 핵심 컨텐츠로 자리하고 있으며, 미래에는 '추천'이라는 개념 자체가 모든 컨텐츠에 있어서 기본으로 깔릴 것으로 예상한다. 

> 대표적으로 ['세이렌'](http://www.slideshare.net/deview/d2-38398255)을 들 수 있겠다. 세이렌은 음악 플레이어에서 '다음 곡' 이 단순히 다음 곡이 아니라 '내게 다음 곡을 추천해봐' 라고 주장한다(물론 셔플일 때 얘기다). 이와 같이 지금은 평범한 '다음'에, '추천'이 녹아들 수 있는 것이다.

이번 기회에 추천 시스템에 대해 자세히 공부하고 정리하도록 한다.

# Recommendation System

추천 시스템에는 크게 두가지 접근이 있다. 아이템의 내용을 분석하여 추천하는 Content-based Approach와 사용자의 평가 내역을 분석하여 추천하는 Collaborative Filtering Approach. 

  * Collaborative Filtering  
좋은 성능. 사용자의 행동 패턴에 따라 적절한 추천.  
그러나 수집된 정보의 양이 많아야 좋은 결과가 나온다. 이를 cold start라 한다.
  * Content-based  
적은 정보만으로도 좋은 추천 가능.  
그러나 모델링 방식에 따라 정확도가 많이 달라지고, 비슷한 아이템 끼리만 추천이 가능하여 추천 범위가 제한된다.

## [Collaborative Filtering](http://ko.wikipedia.org/wiki/%ED%98%91%EC%97%85_%ED%95%84%ED%84%B0%EB%A7%81)

Collaborative Filtering은 아이템을 분석하지 않고 유저의 평가 내역을 이용한다. 이러한 방식의 가장 큰 장점은 복잡한 아이템을 어렵게 분석하지 않아도 된다는 것. 

> 예를 들어 음악 추천 시스템을 위해 음원 분석 알고리즘을 만든다고 해도, 이 음원 분석 엔진의 정확도는 좋아야 60%를 넘기지 못한다. 당연히 영화나 드라마 따위의 동영상 분석은 훨씬 어렵다. 

대신, 아이템이나 유저의 유사도(similarity)를 모델링하고 측정하여 추천한다.

굉장히 널리 쓰이는 CF 시스템이지만 단점도 있는데

  1. Cold start  
유저의 평가 내역이 필요하다. 즉 이러한 데이터가 없는 서비스 초기에는 추천이 부정확함.
  2. Scalability
  3. Sparsity (희소성)

데이터마이닝 시간에 배웠던 Association Rule(연관 규칙) 또한 이 CF 분석에 적용할 수 있겠다.

### User-based

유저 기반 CF는 유저의 행위를 측정 및 분석하고 이를 기반으로 유저간 유사도를 측정한다. 그러면 유사도가 높은, 즉 유사한 유저끼리 해당 유저의 선호 아이템들을 기반으로 추천이 가능하다.

기본적이고 고전적인 방법이다.

### Item-based

그러나 이러한 User-based 방법은 몇가지 문제점이 있다. 먼저 새로운 유저에 대해서는 제대로 된 추천이 불가능하며, 반대로 오랫동안 사용한 유저의 경우 선호도가 분산되기 마련이다. 이러한 문제에 대응하고자 아마존이 사용한 추천 시스템이 바로 이 Item-based CF이다.

아이템 기반 CF는 아이템간의 유사도를 측정하며, 유저가 어떤 아이템을 선호하면 유사한 다른 아이템을 추천할 수 있다.

[아마존닷컴의 개인화 추천](http://worldcup.tistory.com/2460427)

## Content-based

Content-based는 아이템이나 유저를 분석하여 비슷한 아이템을 추천한다. 아이템과 유저간의 액션을 분석하는 것이 아니라 컨텐츠 자체를 분석하기 때문에 많은 양의 유저의 액션을 요구하지 않는다는 것이 장점. 즉 CF의 문제점인 cold start가 없다.

> 예를 들어 삼성과 갤럭시에 대한 뉴스를 보았다고 하면, 이 기사를 분석하여 이 기사의 핵심 엔티티가 '삼성'과 '갤럭시'라는 것을 분석해내고, 이 엔티티를 기반으로 새로운 뉴스를 추천한다.

### TF-IDF

Term Frequency - Inverse Document Frequency. Content-based 분석을 위한 대표적인 방법이다.  
문서에서 어떤 단어가 얼마나 자주 등장하는가(TF), 그 단어가 다른 문서에서도 자주 등장하는가(IDF). TF-IDF 를 이용하여 문서를 수치화 할 수 있다.

Inverse DF인 것은 다른 문서에서도 자주 등장하는 단어라면 중요도가 떨어지기 때문이다. 영어로 따지면 The, That 같은 단어들은 어디에나 등장하기 마련. 물론 이러한 단어는 stopword로서 따로 처리해 줘야 할 필요도 있다.

## RaaS

Recommendation-as-a-Service. Recom.io에서 만든 말인듯?  
[Recom.io](http://recom.io/)  
[Prediction.io](http://prediction.io/)

둘다 완전 무료. 특히 Prediction.io는 오픈소스다. Recom.io는 한국사람들이 만든 서비스인 듯. 물론 회사는 미국에 있다. 한국에서 이렇게 진보적인 서비스를 하다니! 하고 놀랐는데 역시 미국물 먹은 사람들이 만든 거였다.

## 참고

[￼Recommendation System : 협업 필터링을 중심으로](http://rosaec.snu.ac.kr/meet/file/20120728b.pdf) : 카이스트 알고리즘 연구실 자료. 좋은 자료이므로 꼭 읽어보길 추천.

[추천 시스템(Recommender System)](http://dsmoon.tistory.com/entry/%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9CRecommender-system)  
[협응적 추천 시스템(Collaborative Recommender System)](http://dsmoon.tistory.com/entry/CollaborativeRecommenderSystem)  
[내용 기반 추천 시스템(Content-based Recommender System)](http://dsmoon.tistory.com/entry/Contentbased-recommendation)  
개괄적인 개념 잡기에 좋다. 다 같은 블로그.

[MapReduce 기반 대용량 추천 알고리즘 개발](http://readme.skplanet.com/?p=2509) : SK플래닛 기술 블로그. SPADE에서 사용하는 개인화/추천 User-based CF 알고리즘에 대해 살펴보았다. 분산 처리에 대한 설명도 있음.

[Collaborative Filtering : 협업 필터링 추천 방식에 대한 정리](http://www.slideshare.net/springloops/collaborative-filtering-23732558) : slideshare. CF의 간략한 개념과 유사도(Distance) 측정 알고리즘에 대한 설명.


[Tistory 원문보기](http://khanrc.tistory.com/29)
