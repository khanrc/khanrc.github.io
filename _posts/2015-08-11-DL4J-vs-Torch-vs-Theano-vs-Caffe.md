---
layout: post
title: "DL4J vs. Torch vs. Theano vs. Caffe"
tags: ['Deep Learning']
date: 2015-08-11 19:29:00
---
# [DL4J vs. Torch vs. Theano vs. Caffe](http://deeplearning4j.org/compare-dl4j-torch7-pylearn.html)

제목에 나열된 네가지 프레임워크는 가장 핫한 딥러닝 프레임워크들이다. 이 글의 주 내용은 [deeplearning4j](http://deeplearning4j.org/)에서 참고하였으므로, 편향될 수 있다는 점을 고려하자.

Deeplearning4j는 딥러닝 오픈소스 프로젝트들 중에서 그 언어와 목적에서 구분된다. DL4J는 자바 기반이고, 실무에 초점을 맞췄고(industry-based), 커머셜 서포트를 제공하는 대용량의 데이터를 다루도록 디자인된 분산 딥러닝 프레임워크다. GPU를 사용하는 하둡 및 스파크와의 연동을 지원한다.

### Pylearn2/Theano

대부분의 딥러닝 연구는 파이썬으로 작성된 [Pylearn2](http://deeplearning.net/software/pylearn2/)와 [Theano](http://deeplearning.net/software/theano/)기반으로 수행된다. Pylearn2는 머신러닝 라이브러리이고, Theano는 Numpy와 같이 다차원 배열을 다루는 라이브러리다. 이 두 라이브러리는 벤지오가 이끄는 University of Montreal과 LISA group에서 개발한다.

Pylearn2는 MLP, RBMs, Stacked Denoising Autoencoders, ConvNets 등 중요한 알고리즘들을 전부 포함하고 있는 non-distributed 프레임워크다. 이와 비교해서 DL4J는 딥러닝 분야에서 Scikit-learn과 같은 수준을 지향하는데, 즉 parallel GPUs 또는 CPUs를 사용하고 필요하면 하둡이나 스파크와도 연동할 수 있는 자동화된(편리한) 스케일 아웃 방법을 제공하는 것을 목표로 한다.

### Torch

[Torch7](http://torch.ch/)은 Lua로 작성된 머신러닝 프레임워크다. 구글이나 페이스북 등에서 사용한다. Lua는 1990년대 초에 브라질에서 개발된 multi-paradigm language다. Lua는 pure ANSI C로 작성되었기 때문에 C/C++ 라이브러리를 사용하기 쉽다. Pylearn2/Theano를 사용하다 Torch로 옮겨간 사람의 말에 따르면, Torch는 좀 더 native한 non-symbolic programming이 가능하기 때문에 더욱 직관적이고 이해하기 쉽다고 한다.

Torch는 강력하지만 Lua를 기반으로 하기 때문에 접근성이 떨어진다. 또한 사용하기에 쉬운 것 같지는 않다.

> [Why use Torch in facebook and google?](https://news.ycombinator.com/item?id=7929216)   
_Why torch/lua, why not python+?_   
별 이유는 없다. LuaJIT이 강력하며 가볍다. 때문에 작은 기기에도 임베딩할 수 있다. 잘은 모르지만 파이썬으로는 어려울 것이다.   
_Is Torch better than Theano/etc.?_   
장단이 있다. 

### Caffe

[Caffe](http://caffe.berkeleyvision.org/)는 잘 알려지고 널리 사용되는 머신-비전 라이브러리다. 이는 비전 데이터에 특화되어 있어 텍스트나 사운드 또는 타임 시리즈 데이터는 고려하지 않는다. Caffe와 DL4J는 둘 다 최신의 ConvNets image classification 알고리즘을 제공하지만, Caffe는 GPU paraellism을 제공하지 않는다.

### Licensing

Theano/Torch/Caffe: BSD License   
DL4J/ND4J: Apache 2.0 License   
자세히는 모르겠지만 아파치 2.0 라이센스는 BSD 라이센스에 비해 법적 특허 이슈에서 강점이 있다고 한다.

### Speed

DL4J가 사용하는 선형대수 연산을 제공하는 라이브러리인 [ND4J는 Numpy에 비해 두 배 빠르다](http://nd4j.org/benchmarking)고 한다. 또한, Torch7과 DL4J 모두 패러렐리즘을 제공하지만 DL4J는 자동화되어 있다. 즉, 워커 노드와 커넥션을 자동으로 셋팅해준다. 

### Ecosystem

하둡은 자바로 개발되었고, 스파크는 하둡의 Yarn run-time위에서 작동한다. Akka 등의 라이브러리들은 DL4J의 분산 시스템을 구축한다. 

비록 자바가 C나 C++보다는 느리지만 많은 사람들의 생각보다 빠르며, 분산 시스템을 통해 스케일 아웃 할 수 있다.

마지막으로, 우리는 Numpy의 자바 버전을 만들고 있으며 이는 큰 도움이 될 것이다(ND4J가 그 역할인 줄 알았는데 부족한가보다).

### Scala

스칼라는 데이터 사이언스에서 매우 중요한 언어가 되리라 예측하기 때문에, DL4J와 ND4J를 스칼라로 포팅할 계획이 있다. (지금도 이미 스칼라 API를 제공하고 있는 것 같다)

## 요약

  * Theano/Pylearn2: 가장 일반적인 방법
  * Caffe: 비전 데이터에 특화되어 있으며 multi-GPU parallelism을 지원하지 않음
  * Torch7: 직접 내부 코드를 수정하기에 용이 (추측)
  * Deeplearning4j: 위 라이브러리들과는 다르게 인더스트리 타겟으로 개발되어 분산 시스템 지원이 잘 되어 있음

## 참고

[DL4J vs. Torch vs. Theano vs. Caffe](http://deeplearning4j.org/compare-dl4j-torch7-pylearn.html)   
[Popular Deep Learning Tools - a review](http://www.kdnuggets.com/2015/06/popular-deep-learning-tools.html)   
[Best framework for Deep Neural Nets?](https://www.reddit.com/comments/2c9x0s)


[Tistory 원문보기](http://khanrc.tistory.com/117)
