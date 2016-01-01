---
layout: post
title: "[kaggle] Bag of Words Meet Bags of Popcorn - (0) Description"
tags: ['DataScience/Text Mining']
date: 2015-07-08 11:45:00
---
# [Bag of Words Meet Bags of Popcorn - (0) Description](https://www.kaggle.com/c/word2vec-nlp-tutorial)

### 들어가기 전에

kaggle tutorial. 요약번역. 원문의 코드는 python 2지만 여기서는 python 3이다(별 차이는 없다).

원문도 쉽게 되어 있기 때문에 원문을 보는 것도 좋다. 공부하는 겸 해서 옮겨 보았다. 이 튜토리얼을 한번 훑고 나면, kaggle competition에 참가하는 방법도 알 수 있고, nltk와 scikit-learn을 사용하는 python에서의 텍스트 마이닝 기본 과정도 경험할 수 있으며, 딥러닝을 통한 워드 임베딩인 word2vec도 살펴볼 수 있다. 지난번에는 scikit-learn에서 제공하는 20 newspaper tutorial을 해 보았는데 Bags of Popcorn까지 하고 나면 확실히 텍스트 마이닝에 입문할 수 있을 것이다.

무엇보다도, 쉽기 때문에 머리 아플때 보기 좋다!

### Introduction

이 튜토리얼은 감정 분석(sentiment analysis)을 조금 "deeper"하게 살펴본다. 구글의 [Word2Vec](https://code.google.com/p/word2vec/)은 단어의 의미에 초점을 맞춘 멋진 딥러닝 메소드다. Word2Vec은 단어의 의미를 이해하고 단어간의 의미 관계를 파악한다. Word2Vec은 RNN(recurrent neural network)이나 DNN(deep neural nets) 등의 deep approach처럼 작동하지만 그보다 효율적이다. 이 튜토리얼에서는 Word2Vec의 감정분석에 초점을 맞춘다.

감정분석은 머신러닝에서 중요한 주제 중 하나다. 사람들의 표현은 애매모호한 경우가 많기 때문에 사람에게나 컴퓨터에게나 오해하기 쉽다. 영화 리뷰를 감정분석하는 또다른 [캐글 컴페티션](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)가 있다. 이 튜토리얼에서는 이와 비슷한 문제에 Word2Vec을 어떻게 적용하는지를 살펴본다.

[drug discovery](https://www.kaggle.com/c/MerckActivity)나 [cat and dog image recognition](https://www.kaggle.com/c/dogs-vs-cats) 등 딥러닝에 관련된 몇가지 캐글 컴페티션들이 더 있으니 관심 있으면 살펴보도록 하자.

### Tutorial Overview

이 튜토리얼은 두가지 목표로 구성되어 있다:   
**Basic NLP(Natural Language Processing)**: **Part 1**. 기본적인 NLP 프로세싱 테크닉을 다룬다.   
**Deep Learning for Text Understanding**: **Part 2, 3**. Word2Vec을 사용해서 어떻게 모델을 학습하고 결과로 나오는 워드 벡터를 사용해서 어떻게 감정분석을 하는지 자세히 살펴본다.

딥러닝은 아직 정립되지 않은 빠르게 발전하고 있는 분야이기 때문에, 파트 3 또한 정확한 정답이라기 보다 Word2Vec을 사용하는 여러가지 방법을 실험하고 제시한다.

이 튜토리얼에서는 IMDB의 감정분석 데이터셋(IMDB sentiment analysis data set)을 사용한다. 이 데이터셋은 100,000개의 영화 리뷰로 구성되어 있다.

### Data Set

라벨링된 데이터셋은 50,000개의 IMDB 영화 리뷰로 구성되어 있다. 이 리뷰들은 감정분석을 위해 특별히 선택되었다. 리뷰의 감정은 binary로 되어 있는데, IMDB 평점이 5 미만이면 0, 7이상이면 1로 되어 있다. 각 영화들의 리뷰가 30개를 넘지 않으며, 25,000개의 라벨링된 트레이닝셋과 테스트셋은 같은 영화가 전혀 없다. 추가로, 50,000개의 라벨링되지 않은 데이터가 제공된다.

### File descriptions

  * **labeldTrainData** \- 라벨링된 트레이닝 셋. 내용은 tab으로 구분되며 id, sentiment, review로 구성된 header row와 25,000개의 row들이 존재한다.
  * **testData** \- 25,000개의 id, review. 모델을 트레이닝 할 때 테스트셋으로 사용하라는 게 아니라 최종적으로 이 데이터셋을 사용해서 판단한다는 의미로 보인다 (라벨링이 안 되어 있음).
  * **unlabeledTrainData** \- 50,000개의 추가적인 라벨링되지 않은 트레이닝 셋. 
  * **sampleSubmission** \- 제출 포멧.

### Code

[github repo](https://github.com/wendykan/DeepLearningMovies)


[Tistory 원문보기](http://khanrc.tistory.com/98)
