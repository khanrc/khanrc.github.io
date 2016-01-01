---
layout: post
title: "[kaggle] Bag of Words Meet Bags of Popcorn - (4) Part 4: Compare"
tags: ['DataScience/Text Mining']
date: 2015-07-22 15:29:00
---
# Part 4: Comparing deep and non-deep learning methods

## Results

Method | Accuracy  
---|---  
Bag of Words | 0.84380  
Average Vectors | 0.83220  
Bag of Centroids | 0.84216  
  
차이는 거의 없는데, 미세하게 Bag of Words가 제일 좋다.

## Why is Bag of Words better?

가장 큰 이유는, 세 메소드 다 단어의 순서를 무시하는 Bag of Words형태의 피처를 사용했기 때문에 전부 비슷한 결과가 나왔다.

## A few things to try:

먼저, Word2Vec을 더 많은 데이터를 사용해서 트레이닝 하면 더 좋은 성능을 낼 것이다. 구글의 Word2Vec은 10억개의 단어를 트레이닝했지만, 우리 데이터셋에서는 오직 1800만 개를 트레니이 했을 뿐이다. 다행히도, Word2Vec은 트레이닝된 모델을 불러오는 함수를 제공하고 이 함수는 원래 구글이 C로 만들었기 때문에, 마찬가지로 C로 트레이닝한 모델도 파이썬에서 불러올 수 있다.

두번째로, 최근 논문에서 분산 단어 벡터 테크닉(distributed word vector techniques)이 Bag of Words 모델보다 더 좋은 결과를 보였다. 거기에서는 Paragraph Vector라는 알고리즘을 IMDB 데이터셋에 적용하였다. 우리의 접근법과는 다르게 Paragraph Vector는 단어의 순서 정보를 보존한다.


[Tistory 원문보기](http://khanrc.tistory.com/111)
