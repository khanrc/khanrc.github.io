---
layout: post
title: "RL - Distributional RL"
tags: ['RL']
date: 2019-03-23
comments: true
---

* TOC
{:toc}

# Distributional RL

추천 레퍼런스: [RLKorea Distributional RL](https://reinforcement-learning-kr.github.io/2018/09/27/Distributional_intro/)

- Key idea: value function 이 expectation 값을 예측하는 것이 아니라 분포 자체를 예측하게 하자

당연한 얘기지만, 평균값만 아는 것보다 전체 분포를 다 아는 것이 더 좋다. 평균값에 기반한 추론보다 전체 분포에 기반한 추론이 더 정확할 것이다. 이 관점을 우리가 사용하는 value function 에도 적용해볼 수 있는데, value function 의 output 을 평균값 (expectation 값) scalar 로 하는 것이 아니라 전체 분포 (distribution) 로 하면 된다. 즉, value network 가 어떤 state 의 평균 가치를 예측하는 것이 아니라 가치 분포를 예측하도록 하여 더 정확한 추론을 할 수 있다는 것이다. 이러한 관점으로 기존 알고리즘들의 성능을 개선하려는 시도를 distributional RL 이라고 한다.

## C51

Bellemare, Marc G., Will Dabney, and Rémi Munos. "A distributional perspective on reinforcement learning." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.

Value network 가 분포를 예측하도록 변환하는 건 그리 어렵지 않다. 네트워크의 구조를 바꿔주고, loss 를 계산할 때 결과값이 scalar 가 아니라 분포이므로 KL-divergence 를 사용한다. 

마지막으로 target $r+\gamma Q(s,a)$ 만 distributional 하게 바꾸면 된다. 아래 그림을 참고하자. 수식 그대로 연산을 수행하고 처음에 정해놓은 분포 규격으로 맞춰주는 과정을 거친다.

![c51]({{site.url}}/assets/rl/dqn-c51.png)
