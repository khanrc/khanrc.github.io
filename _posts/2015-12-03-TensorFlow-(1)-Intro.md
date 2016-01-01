---
layout: post
title: "TensorFlow - (1) Intro"
tags: ['DataScience/Deep Learning']
date: 2015-12-03 01:35:00
---
# [TensorFlow](http://www.tensorflow.org/)

텐서플로는 굉장히 흥미로운 내부 구조를 가지고 있다. Theano나 Torch도 마찬가지의 구조를 차용한다고 하는데, 나는 텐서플로에서 처음 접했다. 프로그래밍 언어 패러다임 중에 Stream Programming 이라는 것이 있다. 이 패러다임은 프로그램을 flow graph 형태로 구조화하고, 각 노드는 데이터를 받아 처리하여 출력한다. 데이터는 엣지를 따라 흐른다. 이 패러다임에 따라 구조화된 프로그램은 유저가 특별히 신경쓰지 않아도 내부적으로 (자동적으로) 패러렐이 가능하다. 

![parallel stream graph](http://publications.csail.mit.edu/abstracts/abstracts07/mgordon/mgordon1.jpg)

대표적인 스트림 프로그래밍 언어로는 [StreamIt](http://publications.csail.mit.edu/abstracts/abstracts07/mgordon/mgordon.html) 이 있다. 텐서플로는 파이썬 인터페이스를 제공하지만, 내부적인 실행 매커니즘은 이 스트림 프로그래밍에 기반한다. 기존에 파이썬이 제공하던 imperative programming 과는 다르게, 코드 한줄 한줄이 그 즉시 실행되지 않고 컴퓨테이션 그래프를 생성하며, 나중에 그래프를 다 생성한 후에 이 그래프를 통째로 실행한다. 그러면 내부적으로 패러렐하게 실행된다. 

> 정확하게는, 코드 한줄한줄은 당연히 그 때 실행된다. 단지 그 실행이 연산을 하는 것이 아니라 그래프에 노드를 추가하는 작업이라서 실제로 연산이 되지는 않는다는 것이다. 더 자세한 건 Basic Usage를 참고하자.

## Introduction

[Google + open-source = TensorFlow](http://www.datasciencecentral.com/profiles/blogs/google-open-source-tensorflow) 를 요약번역한 것. 별 내용은 없다.

### What is TensorFlow?

머신러닝 라이브러리. [마이크로소프트의 Azure ML](https://azure.microsoft.com/en-us/services/machine-learning/) 같은 클라우드 기반 머신러닝 서비스가 아님. "라이브러리" 다.

### What is flow graph?

![flow graph](http://nlpx.net/wp/wp-content/uploads/2015/11/TensorFlow-graph1.jpg)

이러한, 말 그대로 flow graph인데, 텐서플로우에는 이걸 자동으로 그려주는 `TensorBoard` 라는 모듈이 있음.

### What about tensors?

텐서플로우에서 데이터는 텐서로 표현되고, 텐서는 다차원의 다이나믹 사이즈 데이터 어레이다 (multidimensional and dynamically sized data array). 이 텐서가 플로우 그래프에서 흐르게 (flow) 되므로 tensor flow가 이 라이브러리의 이름인 것.

### What is cool about TensorFlow?

  1. 표현의 유연성 - 알고리즘을 데이터 플로우 그래프로 나타낼 수 있으면 텐서플로우로 구현할 수 있다.
  2. CPU와 GPU 를 모두 사용하며 parallel &amp; asynchronous 컴퓨팅을 지원한다.
  3. 연구(리서치)와 실무(프로덕션) 모두에서 사용할 수 있다.
  4. 자동 미분 (auto-differentiation). 텐서플로우는 도함수 (derivative) 를 자동으로 계산하고 이는 매우 편리하다. 특히 네가 gradient-based 알고리즘을 사용한다면 - SGD 같은거.
  5. 파이썬 인터페이스! 

### Licencse

Apache 2.0 - 연구와 상업적 목적으로 전부 프리하게 쓸 수 있음.

### Etc

본문에 보면 튜토리얼 및 예제도 소개하고 있음. 특히, 텐서플로우의 scikit-learn 식 인터페이스를 제공하는 [Scikit Flow](https://github.com/google/skflow) 는 흥미롭다! 또한 Theano와의 비교도 간략하게 적혀 있으니 참고하자.


[Tistory 원문보기](http://khanrc.tistory.com/132)
