---
layout: post
title: "검색엔진 (2) - 라이브러리: Lucene, Solr, Elasticsearch"
tags: ['그 외/Tech']
date: 2014-08-26 01:50:00
---
# 검색엔진 라이브러리

검색엔진 라이브러리에 대해 알아본다. lucene, solr, elasticsearch.

## [Lucene](http://linuxism.tistory.com/898)

[검색이야기 with Lucene, solr](http://zeous.egloos.com/1412280)

검색 프로세스는 기본적으로

  1. 데이터 수집(웹 스파이더링) - 50%
  2. 데이터 인덱싱 - 40%
  3. 데이터 검색 - 10%

로 나뉘어진다.

루씬은 이 과정에서 인덱싱과 검색의 API를 제공하는 코어 엔진이다. 원래 Java로 개발되었으며 현재는 Perl, Python, C++, php 등 다양한 언어로 포팅되어 있다.

#### 참고

##### 한국어

[최근 lucene-Korean-Analyzer 프로젝트의 근황](http://devyongsik.tistory.com/655)  
[루씬기반인 한국어 형태소분석기를 제공하는 다봇](http://dabot.tistory.com/)  
역시 오픈소스 라이브러리를 이용하여 프로젝트를 하려면 항상 한국어가 걸림돌이다. 위 링크를 참고하자.

##### 스핑크스(Sphinx)

[검색엔진 루씬(Lucene)과 스핑크스(Sphinx) 소개](http://blog.jidolstar.com/863)  
[검색엔진 스핑크스 Sphinx 도입](http://iramine.com/34)  
[Sphinx 사소한 팁 몇가지](http://jong10.com/post/30221511694)  
스핑크스. 루씬보다 빠르고, 적용이 간단하다고 한다. 

## Solr

[Joinc: Solr로 로컬 검색서비스 만들기](http://www.joinc.co.kr/modules/moniwiki/wiki.php/Site/Search/Document/Solr)  
루씬을 한번 더 래핑한 오픈소스. 인덱싱과 검색은 루씬엔진을 사용하고, http 통신 및 관리툴을 제공한다. 

**참고**  
[[SOLR 강좌] SOLR 소개](https://lael.be/594)  
[Ubuntu install SOLR KOR(SOLR 설치하기 한글 형태소분석기)](https://lael.be/592)

## Elasticsearch

[elasticsearch로 로그 검색 시스템 만들기](http://helloworld.naver.com/helloworld/273788)  
솔라와 마찬가지로 루씬 기반의 검색엔진. 멘토님의 말씀도 그렇고 전반적으로 솔라가 더 평가가 좋은 듯 싶다.

## Solr vs Elasticsearch

[Solr vs ElasticSearch](http://jeen.github.io/2013/07/15/solr-vs-elasticsearch-part-1/) : 무려 6개의 포스팅으로 구성된 장문의 글.  
[Solr vs. ElasticSearch](http://stackoverflow.com/questions/10213009/solr-vs-elasticsearch) : 스택오버플로. 당연히 영어!  
[System Properties Comparison Elasticsearch vs. Solr vs. Sphinx](http://db-engines.com/en/system/Elasticsearch%3BSolr%3BSphinx)


[Tistory 원문보기](http://khanrc.tistory.com/28)
