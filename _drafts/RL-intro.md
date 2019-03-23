---
layout: post
title: "RL - Introduction"
tags: ['RL']
date: 2019-03-18
comments: true
---

# Reinforcement Learning

## 들어가며

이 시리즈는 OpenAI 에서 나온 spinning up 에 소개된 key papers 를 중심으로 최근 RL 에서 연구되고 있는 여러 분야들을 알아봅니다. 논문을 디테일하고 엄밀하게 살펴보기보다는 각 논문에서 제안하는 핵심적인 아이디어들을 직관적으로 설명하는 것에 중점을 두었습니다. 이후의 내용들은 편의상 평어로 작성되었습니다.

## Overview

![taxonomy](/assets/rl/intro-rl-algo-taxonomy.svg)

모든 분류가 그렇듯이 RL 알고리즘도 위와 같은 트리 구조로 분류하기에는 어려움이 있다. 

Spinning up 에서 소개하고 있는 모든 [key papers](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) 를 여기서 다루는 것은 아니며, 반대로 여기에서 소개하는 논문이 key papers 에 없는 경우도 있다. 이 시리즈에서 다룰 논문들은 다음과 같다:

참고)

- 아래 논문 리스트는 확정된 것이 아니며, 작성 과정에서 수정될 수 있음
- 일부 2번씩 등장하는 논문들이 있는데, 추후 해당 섹션까지 글이 작성될 때 분류를 재조정할 예정인 논문들임

1. Deep Q-learning
    - DQN (Deep Q-Networks)
    - DDQN (Double DQN)
    - Dueling DQN
    - PER (Priortized Experience Replay)
    - C51
    - NoisyNet
    - Rainbow
1. Policy gradients
    - REINFORCE
    - Actor-Critic
    - Off-policy Actor-Critic
    - A3C (Asynchronous Advantage Actor-Critic)
    - GAE (Generalized Advantages Estimation)
    - NPG (Natural Policy Gradients)
    - TRPO (Trust Region Policy Optimization)
    - PPO (Proximal Policy Optimization)
1. Deterministic policy gradients
    - DPG (Deterministic Policy Gradients)
    - DDPG (Deep DPG)
    - TD3 (Twin delayed DDPG)
1. Distributional RL
    - ~~C51~~
1. Path-Consistency Learning
    - PCL
1. Scaling RL
    - ~~A3C~~
    - IMPALA (Importance Weighted Actor-Learner Architecture)
    - Ape-X
1. Exploration
    - Learnable exploration
        + ~~NoisyNet~~
    - Entropy regularization
        + SAC (Soft Actor-Critic)
        + TAC (Tsallis entropy Actor-Critic)
    - Intrinsic motivation
        + CTS
        + RND (Random Network Distillation)
1. Transfer and Multitask RL
    - ~~HER~~
    - PathNet
1. Hierarchical RL
    - HER (Hindsight Experience Replay)
    - HIRO (Hierarchical RL with Off-policy correction)
    - STRAW (Strategic Attentive Writer)
1. from Demonstration
    - DDPGfD
    - DQfD
    - Youtube
1. Model-based RL
    - I2A (Imagenation-Augmented Agent)
    - World model
    - AlphaGoZero / AlphaZero / EXIT (Expert Iteration)
1. Meta-RL
    - RL^2
    - SNAIL (Simple Neural Attentive Learner)
1. IRL
    - GAIL (Generative Adversarial Imitation Learning)

DQN 이나 REINFORCE 등 시작점이 되는 알고리즘들은 간단하게 다루기는 하나 기본적으로 알고 있다고 가정한다.

<!-- Not covered here:

1. Memory
2. Safety
3. Reproducibility, Analysis, and Critique -->
