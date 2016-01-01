---
layout: post
title: "Generative model vs Discriminant model"
tags: ['DataScience']
date: 2015-07-13 15:56:00
---
# Generative model vs Discriminant model

## 개념

  1.   2.   3. (세 수식 다 같은 식이다)

Generative model은 데이터를 생성(generate)하는 모델의 관점이다. 확률적으로 likelihood와 prior에 대한 모델이라고도 하는데, 이것이 결국 데이터를 생성하는 관점이다. 위 식을 보자. 1번 식에서 likelihood와 prior는 각각 와 인데, 이는 결국 얼룩말이 이미지로 변환되는 (즉 데이터가 생성되는) 과정을 의미한다. 2번 식에서의 ,  또한 문서의 주제를 정하고 그 주제로부터 단어가 생성되는 과정을 의미한다. 즉, Generative model은 데이터를 기반으로 이 데이터를 생성하는 모델을 추정하여 해당 데이터의 클래스를 판별해낸다고 할 수 있다.

반면, Discriminant model은 데이터 자체의 차이를 판별한다. 즉, 해당 데이터가 어떻게 생성되는지는 관심이 없고 데이터간의 차이를 학습하는 것이다. 이는 posterior에 대한 모델이라고 하는데, 결국 posterior라 함은 데이터가 있을 때 어느 클래스에 속하냐를 의미한다. 이는 데이터간의 차이를 학습하므로 decision boundary를 만드는 접근법이기도 하다.

## 예시

  1. 임의의 문장이 어떤 언어인지를 판별하는 문제라고 하자.   
1) 모든 언어를 학습해서 그 지식을 기반으로 판별하는 것이 Generative approach이고,   
2) 언어를 학습하는 것이 아니라 문장들의 언어적 차이를 통해 판별하는 것이 Discriminant approach 이다.

  2. 어떤 시그널을 카테고라이징 한다고 하자.   
Generative algorithm: 데이터가 어떻게 생성되는지를 모델링한다. generation assumption에 기반해서, 어떤 카테고리가 이 시그널을 생성할 확률이 제일 높은가?   
Discriminative algorithm: 데이터가 어떻게 생성되었는지에 상관 없이, 단순히 이 데이터만을 보고 카테고라이징 한다.

## 참고

[stackoverflow; What is the difference between a Generative and Discriminative Algorithm?](http://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm)   
베스트 답변 외에 좋은 답변들이 많으니 쭉 살펴보자. 위 예시도 여기서 가져왔다.


[Tistory 원문보기](http://khanrc.tistory.com/104)
