---
layout: post
title: "Distance Metric Learning"
tags: ['DataScience']
date: 2014-10-16 11:37:00
---
# [Distance Metric Learning](http://sanghyukchun.github.io/37/)

머신러닝에서 디스턴스 함수가 중요한 알고리즘들이 있다. 대표적으로 kNN. 이러한 디스턴스 함수를 정의하는 분야가 따로 있는데, Distance Metric Learning이라 한다. 몇가지 [대표적인 규칙](http://en.wikipedia.org/wiki/Metric_\(mathematics\))이 있다. 

## Distance Metric

  * Euclidean distance

\\(d(p,q)=\sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + ...}\\)

데이터의 `correlation`을 고려하지 못한다.

  * Mahalanobis distance

\\(d(p,q)=\sqrt{(\vec p - \vec q)^\top \Omega (\vec p - \vec q)}\\)

`Mahalanobis distance`는 데이터셋의 `correlation`을 알아서 처리한다. `Ω`가 `covariance matrix`이고 따라서 이 metric이 데이터의 `correlation`을 포함하여 거리를 표현한다. 그러나 임의의 데이터에 대해 `covariance matrix`를 계산하는 것은 매우 어려우며 이를 위해 LMNN classification등 `Ω`를 learning하는 method들도 존재한다.

## Distance Metric Learning

  * Supervised Learning  
NCA (Neighbourhood Components Analysis), RCA (Root Cause Analysis)
  * Unsupervised Learning  
[PCA (Principal component analysis)](http://ko.wikipedia.org/wiki/%EC%A3%BC%EC%84%B1%EB%B6%84_%EB%B6%84%EC%84%9D)
  * Maximum margin based Learning  
LMNN
  * Kernel Method

위 두개가 일반적인 분류. 위에서 언급한 바와 같이 메트릭을 러닝한다. `PCA`는 dimension reduction을 위한 대표적인 메소드이며 이와 같이 데이터 디스턴스를 측정하기 이전의 전처리 과정에서 디스턴스 측정의 정확도를 높이기 위해 행한다. 


[Tistory 원문보기](http://khanrc.tistory.com/58)
