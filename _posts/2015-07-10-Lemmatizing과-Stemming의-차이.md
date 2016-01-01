---
layout: post
title: "Lemmatizing과 Stemming의 차이"
tags: ['DataScience/Text Mining']
date: 2015-07-10 16:17:00
---
# Lemmatizing과 Stemming의 차이

텍스트 마이닝의 프리프로세싱 단계에서 같은 단어를 하나로 묶어주는 작업을 해야 한다. 대소문자를 통일해 주고, 구두점을 제거해 주고, 진행형이나 복수형 등 다양한 형태로 나타나는 단어들을 묶어준다. 물론 이 과정에서 다른 의미로 쓰인 단어가 묶일 수도 있는데 그건 어쩔 수 없다. 임플레멘테이션 하는 사람이 선택해야 하는 문제다. 예를 들자면, "messages", "messaging", "message" 이 세 단어들은 전부 다르지만 텍스트마이닝의 특성으로 사용할 때는 하나로 묶어주는 것이 바람직하다. 여기서 다양한 형태로 발현된 단어들을 하나로 묶어주는 작업을 stemming과 lemmatizing이라고 한다. 이 둘은 비슷하지만 조금 다르다.

[Lemmatisation 위키](https://en.wikipedia.org/wiki/Lemmatisation)에서 확인할 수 있다. stemming은 문맥이나 어떤 데이터베이스를 이용하지 않고 간단한 룰 기반으로 수행되고, lemmatizing은 더 많은 정보를 고려하여 수행된다. 즉, stemmer가 lemmatizer에 비해 빠르지만 더 낮은 정확도를 보인다.

예를 들자면:   
1\. "better"과 "good"은 같은 lemma(단어의 기본형)를 가지고 있다. 이 연결(link)은 stemming으로는 찾을 수 없고 사전을 뒤져보는 작업이 필요하다.   
2\. "walk"와 "walking"의 연결은 stemming과 lemmatizing 둘 다 잘 찾아낸다.   
3\. "meeting"의 경우, 문맥에 따라 명사일 수도 있고 동사일 수도 있다. 예를 들어 "in our last meeting"에서는 명사고, "We are meting again tomorrow"에서는 동사다. stemming과는 다르게 lemmatizing은 문맥을 고려해서 이를 구분할 수 있다.

[stackoverflow: what is the true difference between lemmatization vs stemming?](http://stackoverflow.com/questions/1787110/what-is-the-true-difference-between-lemmatization-vs-stemming) 참고.


[Tistory 원문보기](http://khanrc.tistory.com/102)
