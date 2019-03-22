---
layout: post
title: "RL - Deterministic policy gradients"
tags: ['RL']
date: 2019-03-23
comments: true
---

* TOC
{:toc}

# Deterministic policy gradients

Policy gradients 는 기본적으로 stochastic policy $\pi(a\|s)$ 를 다룬다. 하지만 대부분의 optimal policy 는 deterministic policy $a=\mu(s)$ 다. 어떤 상황에 정답인 선택지가 딱 정해져 있지, 랜덤하게 두 선택지 중 확률적으로 고르는 것이 optimal 인 경우는 거의 없다는 얘기다. Deterministic policy gradients 계열은 이러한 문제인식으로부터 출발하여 stochastic policy 가 아니라 deterministic policy 를 최적화한다.

## DPG

Silver, David, et al. "Deterministic policy gradient algorithms." ICML. 2014.

- Key idea: 대부분의 task 에서 optimal policy 는 deterministic policy 이므로, 처음부터 deterministic policy 를 target policy 로 두고 학습하자

DPG 는 처음으로 위의 문제를 제기하고 deterministic policy 를 제안한 논문이다. Deterministic policy 를 쓸 경우 expectation 을 취할 때 state 와 action 모두에 대해서 해야 하는 stochastic policy 와는 달리 state 에 대해서만 취하면 되므로 sample efficiency 가 좋아진다.

DPG 에서는 이렇게 deterministic policy 를 취했을 때 이에 대한 deterministic policy gradient 를 구한다:

$$
\nabla_\theta J(\mu_\theta) = \mathbb E_{s\sim \rho_\mu}\left[ \nabla_\theta Q^\mu(s,\mu_\theta(s)) \right]
$$

위 deterministic policy gradient 는 꽤나 직관적인 수식으로, 각 state 에 대해 Q-function 의 기댓값을 최대화하도록 policy update 가 이루어진다. PG theorem 과 같이, state distribution 에 대한 gradient 를 구할 필요가 없다는 것이 DPG theorem 의 핵심이 되며, 증명 과정도 거의 유사하다.

Policy 가 deterministic 이 되면 stochastic policy 에서 자연스럽게 되던 exploration 이 되지 않는 문제가 생긴다. 따라서 DPG 에서는 Q-learning 처럼 노이즈를 주어 exploration 을 수행하는 behavior policy 를 따로 두며, 이를 위한 off-policy DPG 는 다음과 같다:

$$
\nabla_\theta J(\mu_\theta) = \mathbb E_{s\sim \rho_b}\left[ \nabla_\theta Q^\mu(s,\mu_\theta(s)) \right]
$$

위 식과 비교하면 단지 state distribution 이 behavior policy 로 바뀌었을 뿐이다. 위에서 언급했듯 DPG 가 deterministic 한 특성 때문에 action 에 대해서는 expectation 을 취하지 않는데, 이 덕분에 off-policy DPG 에서도 importance sampling 이 필요 없게 된다.

## DDPG

Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).

- Key idea: DPG + DQN Critic

앞서 살펴보았듯 DPG 는 off-policy 방법이기 때문에 같은 off-policy 알고리즘인 DQN 의 트릭들을 가져다 사용할 수 있다. DDPG 는 DQN 의 experience replay 와 freeze target network 를 그대로 가져다 적용한다. 그리고 여기에 추가로 target network 를 업데이트 할 때 주기적으로 한번에 업데이트하던 DQN 과는 달리 지속적으로 조금씩 업데이트하는 soft target update 방식을 제안하여 학습의 안정성을 높였다.

$$
L_Q(\phi)=\mathbb E_D \left[ \left( Q_\phi(s,a)-(r+\gamma Q_{\phi^-}(s',\mu_{\theta^-}(s'))) \right)^2 \right]
$$

## TD3

Fujimoto, Scott, Herke van Hoof, and Dave Meger. "Addressing function approximation error in actor-critic methods." arXiv preprint arXiv:1802.09477 (2018).

TD3 은 Twin Deplayed DDPG 의 약자로, DDQN 에서 다루었던 Q-learning 의 overestimation bias 를 해결하기 위한 twin Q-function 과 variance reduction 을 위한 delayed policy update 를 제안한다.

DQN 에서 발생하는 overestimation bias 를 해결하기 위해 DDQN 에서 double Q-learning 을 도입했었다. TD3 에서는 이러한 bias 가 DDPG 에서도 마찬가지로 발생함을 보이고, 이를 해결하기 위해 Clipped double Q-learning 을 제안한다.

$$
y=r+\gamma \min_{\text{i=1,2}} Q_{\phi_i^-}(s',\mu'(s'))
$$

- DQN (and Q-learning) 에 overestimation problem 이 있듯이, DPG 에도 approximate Q 를 사용하면서 생기는 overestimation problem 이 있음.
  - **Clipped double Q-learning**
    - double Q-learning + underestimation technique
    - 2개의 Q-function 을 사용하고, 실제로 target Q 를 계산할 때에는 두 Q값 중 낮은 값을 사용한다.
- Variance reduction
  - **Target policy smoothing**
    - Target policy 에 적당히 노이즈를 섞어서 스무딩시킴
    - Entropy regularization 처럼 regularizer 역할. 어쩌다 값이 튀어서 peak 한 Q 값이 나와도 적당히 스무딩시켜서 너무 크게 영향받지 않도록 해줌
  - **Soft target update & Delayed policy update**
    - Soft target update: 네트워크를 조금씩 업데이트해줘서 variance reduction (DDPG 와 마찬가지)
    - Delayed policy update: critic 을 actor 보다 더 많이 학습시킴 (critic 이 정확해야 policy 도 잘 학습됨)