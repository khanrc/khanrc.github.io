---
layout: post
title: "Semi-supervised, Weakly-supervised"
tags: ['DataScience']
date: 2015-03-25 17:52:00
---
# Semi-supervised, Weakly-supervised

<http://cs.stackexchange.com/questions/2907/what-exactly-is-the-difference-between-supervised-and-unsupervised-learning>

Semi-supervised와 weakly-supervised가 헷갈려서 찾아보았다.

  * **Supervised**: fully labelled
  * **Semi-supervised**: labelled+unlabelled
  * **Weakly-supervised**: Bootstrapping 또는 self-training이라고도 불림. 적은 수의 샘플에서 시작해서, classifier를 만들고, positive example을 예측해서 labeling을 하고 다시 re-training을 함으로써 classifier를 성장시킨다. 당연한 얘기지만 positive example 예측이 잘못될 경우 classifier가 더 나빠질 수 있다.
  * **Unsupervised**: no labelled


[Tistory 원문보기](http://khanrc.tistory.com/89)
