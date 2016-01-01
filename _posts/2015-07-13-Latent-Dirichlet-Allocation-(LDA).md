---
layout: post
title: "Latent Dirichlet Allocation (LDA)"
tags: ['DataScience/Text Mining']
date: 2015-07-13 14:33:00
---
# Latent Dirichlet Allocation (LDA)

LDA는 문서(도큐먼트)의 주제(토픽)를 찾는 Generative Model이다. 문서가 가질 수 있는 주제들에 대한 확률분포를 알고, 각 주제에서 나올 수 있는 단어들의 확률분포를 알 때, 랜덤 프로세스에 의해 문서를 생성할 수 있다고 가정한다. 이 가정 하에서, 반대로 문서들이 있으면 이 확률분포들을 추정하여 주제를 찾는 방법이다. 이 때 각 주제들이 나올 확률이 디리클레 분포를 따른다는 가정을 하기 때문에 이름이 디리클레 할당이다.

## Introduction

  * I like to eat broccoli and bananas.
  * I ate a banana and spinach smoothie for breakfast.
  * Chinchillas and kittens are cute.
  * My sister adopted a kitten yesterday.
  * Look at this cute hamster munching on a piece of broccoli.

이런 문장들이 있을 때, 여기서 문장의 **토픽**을 찾아내는 것이 바로 LDA다. 예를 들어 이 문장들을 놓고 2개의 토픽을 찾으라고 하면 LDA는 아래와 같이 수행한다:

  * **Sentences 1 and 2**: 100% Topic A
  * **Sentences 3 and 4**: 100% Topic B
  * **Sentence 5**: 60% Topic A, 40% Topic B
  * **Topic A**: 30% broccoli, 15% bananas, 10% breakfast, 10% munching, … (at which point, you could interpret topic A to be about food)
  * **Topic B**: 20% chinchillas, 20% kittens, 20% cute, 15% hamster, … (at which point, you could interpret topic B to be about cute animals)

자, 그럼 LDA는 이것을 어떻게 찾을까?

## LDA Model

LDA는 확률에 기반하여 도큐먼트를 **토픽들(mixture of topics)**로 표현한다. 이 방법은 우리가 도큐먼트를 작성할 때에 대한 몇 가지 가정이 들어간다: 

  * 푸아송 분포(Poisson distribution)에 따라, 도큐먼트에 들어갈 N개의 단어를 결정.
  * K개의 토픽 셋에서 디리클레 분포(Dirichlet distribution)에 따라, 도큐먼트의 토픽들을 결정. 예를 들어, 위 예제에서 food와 cute animal의 두가지 토픽이 있다고 하면, 도큐먼트는 1/3 food와 2/3 cute animal로 구성되도록 선택되었을 것이다.
  * 도큐먼트의 각 단어 $w_i$를 선택한다:   

    * 먼저 토픽을 고른다: 위에서 조사한 다항분포(multinomial distribution)에 따라. 예를 들어, food와 cute animal을 1/3과 2/3으로 선택하였다.
    * 이 토픽을 사용해서, 토픽의 다항분포에 따라 단어를 생성한다. 예를 들어, food 토픽을 선택했다면, "broccoli"는 30%, "bananas"는 15% 등의 확률로 생성한다.

이와 같이 도큐먼트들에 대해 generative model을 가정하고, LDA는 거꾸로 도큐먼트로부터 가장 그럴 확률이 높은 토픽을 찾아낸다.

> Generative Model이란, 어떤 확률분포와 파라메터가 있을 때 그로부터 랜덤 프로세스에 의해 데이터를 생성하는 관점의 모델이다. 즉 여기서는 도큐먼트를 모델링하는 토픽과 토픽별 단어 생성 확률분포를 안다면, 이를 통해 도큐먼트의 생성 확률을 추정할 수 있고 반대로 도큐먼트를 알 때 확률 분포를 추정할 수 있다.

## Learning

자 이제 우리가 도큐먼트 셋을 갖고 있다고 해 보자. 먼저 찾고 싶은 토픽의 개수 K를 정하고, 이제 LDA로 각 도큐먼트들의 토픽을 찾고 토픽에 해당하는 단어들을 찾고 싶다. 어떻게 해야 할까? 여러가지가 있지만 collapsed Gibbs sampling이라고 알려진 방법을 소개한다:

  * 각 도큐먼트에, K개의 토픽들 중 하나를 랜덤하게 할당한다.
  * 이제 모든 도큐먼트들은 토픽을 갖고, 모든 토픽은 단어 분포를 갖게 되었다(물론 잘못되었지만). 이제 이를 개선하자.
  * 각 도큐먼트 d에 대해:   

    * d의 각 단어 w에 대해:   

      * 각 토픽 t에 대해, 다음 두가지를 계산한다:   

        1. p(topic t | document d): 도큐먼트 d의 단어들 중 토픽 t에 해당하는 단어들의 비율을 계산한다.
        2. p(word w | topic t): 단어 w를 갖고 있는 모든 도큐먼트들 중 토픽 t가 할당된 비율을 계산한다. 
      * 이후 p(topic t | document d) * p(word w | topic t)에 따라 토픽 t를 새로 고른다. 우리의 generative model에 따라, 이것은 토픽 t가 단어 w를 생성할 확률이고, 따라서 현재 단어들(도큐먼트들)의 토픽을 이 확률에 따라 다시 설정하는 것이 합당하다.
      * 즉, 이 스텝은, 지금 측정하고 있는 단어 w 외에 다른 단어들이 전부 알맞게 할당되었다고 가정하고, 확률을 계산하여 현재 단어 w를 업데이트한다.
  * 위 스텝들을 충분히 반복하고 나면, 안정적인 상태에 도달한다. 

> 원문에도 적혀 있지만 위 내용은 생략된 부분이 많음(특히 priors/pseudocounts의 사용에 대해). 적당히 넘어가자. 아무튼 이 과정은 마치 클러스터링과 비슷하다. 적당히 할당하고 에러를 계산, 재할당을 반복한다.

## 참고

[Introduction to Latent Dirichlet Allocation](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/): 이 글의 원문이다. 다른 도메인에서의 설명도 있으니 살펴보자.   
[LDA(Latent Dirichlet Allocation): 겉핥기](http://www.4four.us/article/2010/11/latent-dirichlet-allocation-simply): 상대적으로 수식적인 측면에서 접근한다. 가 보면 깁스 샘플링을 통한 파라메터 추정 또한 볼 수 있다. 참고하면 도움이 많이 될 것이다.   
[텍스트의 통계학: (3) 네 주제를 알라](http://nullmodel.egloos.com/1958448): 어려운 부분을 다 빼고 쉽게 설명한다. 더 쉬운 설명이 필요하다면 참고하자.

이 외에 [한국어 위키](https://ko.wikipedia.org/wiki/%EC%9E%A0%EC%9E%AC_%EB%94%94%EB%A6%AC%ED%81%B4%EB%A0%88_%ED%95%A0%EB%8B%B9)도 굉장히 자세히 설명되어 있는데 너무 어렵다!


[Tistory 원문보기](http://khanrc.tistory.com/103)
