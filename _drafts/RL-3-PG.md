---
layout: post
title: "RL - Policy gradients"
tags: ['RL']
date: 2019-03-18
comments: true
---

# Policy gradients

추천 레퍼런스: 
- https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
- [RLKorea PG여행](https://reinforcement-learning-kr.github.io/2018/06/29/0_pg-travel-guide/)

Policy gradient theorem 을 통해 얻을 수 있는 (vanilla) policy gradient 수식은 다음과 같다:

$$
\nabla_\theta J(\pi_\theta)=\mathbb E_{\tau\sim \pi_\theta} \left[ \sum^T_{t=0} A^{\pi_\theta} (s_t, a_t) \nabla_\theta \log \pi_\theta (a_t|s_t) \right]
$$

여기서 여러가지 변주를 줄 수 있는데, GAE 논문에 잘 정리되어 있다.

<div class="imgc">
![pg-gae]({{site.url}}/assets/rl/pg-gae.png){:width="80%" .center}
</div>

여기서 1, 2번이 REINFORCE 에 해당하고, 3번이 REINFORCE with baseline, 4, 5, 6번이 Actor-Critic 에 해당한다.

## REINFORCE

Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.

REINFORCE 는 위 피규어에서 1, 2, 3 에 해당하는 방법론이다. 이 세가지 수식이 전부 같은 expectation 값을 갖는데, 앞쪽 수식일수록 variance 가 크다. Q-learning 계열도 마찬가지지만 PG 계열에서도 expectation 을 sampling 으로 대체하게 되는데, 여기서 발생하는 variance 를 잡는 것이 주요한 챌린지가 되며, REINFORCE 에서도 variance 를 줄이기 위한 노력들을 엿볼 수 있다.

$$
\nabla_\theta J(\pi_\theta)=\mathbb E_{\tau\sim \pi_\theta} \left[ \sum^T_{t=0} (G_t-b(s_t)) \nabla_\theta \log \pi_\theta (a_t|s_t) \right]
$$

여기서 $G_t$ 는 timestep t 에서의 expected return, $b(s_t)$ 는 baseline 에 해당한다. REINFORCE 는 return G 를 알아야 하기 때문에 하나의 에피소드가 끝나야만 학습을 수행할 수 있다.

## Actor-Critic

Actor-critic 에서는 return G 를 Q-function 으로 estimate 하고, bootstrapping 을 통한 학습을 함으로써 에피소드가 끝나지 않아도 학습이 가능해진다.

![pg-ac-alg](/assets/rl/pg-ac-alg.png){: width="80%" .center}

Actor-critic 은 이러한 Q actor-critic 외에도 여러 종류가 있다:

![pg-actor-critic]({{site.url}}/assets/rl/pg-ac.png){:width="80%" .center}
*Image taken from CMU CS10703 lecture slides*

Critic 으로 advantage function A(s,a) 를 사용하는 advantage actor-critic 이 바로 A2C 다. 여기서 multiple agent 를 두고 업데이트를 asynchronous 하게 수행하는 A3C 로 발전한다.

## Off-policy Actor-Critic

Degris, Thomas, Martha White, and Richard S. Sutton. "Off-policy actor-critic." arXiv preprint arXiv:1205.4839 (2012).

- Key contribution: off-policy policy gradient 유도

PG 계열 알고리즘들은 기본적으로 on-policy 다. Expectation 항을 보면 trajectory $\tau$ 가 current policy $\pi_\theta$ 에서 sampling 되기 때문에 과거의 trajectory 를 사용할 수 없다. Off-policy actor-critic 에서는 importance sampling 을 이용하여 off-policy policy gradient 를 유도하고, 이것이 근사식임에도 local optima 로 수렴한다는 것을 증명한다.

$$
\nabla_\theta J_b(\pi_\theta)=\mathbb E_{\tau\sim \pi_b} \left[ \frac{\pi_\theta(a|s)}{b(a|s)} Q^{\pi_\theta}(s,a) \nabla_\theta \log \pi_\theta(a|s) \right]
$$

## A3C

Mnih, Volodymyr, et al. "Asynchronous methods for deep reinforcement learning." International conference on machine learning. 2016.

A3C 는 Asynchronous Advantage Actor-Critic 로, A2C 의 parallel 버전이다. 여러 쓰레드에 각각 actor 를 두고 동시에 simulation 을 하고, 각 액터는 gradient 를 계산해서 누적하여 갖고 있다가 주기적으로 policy network 와 value network 를 업데이트 해 준다.

![pg-a3c](/assets/rl/pg-a3c.jpg){:width="70%" .center}

재미있는 점은 이 때 동기화 과정이 없다는 것이다. 원래대로라면 Global network 를 갖고 있는 마스터 노드에서 각 그라디언트를 취합하여 네트워크를 업데이트하고, 다시 이 네트워크를 각 액터 쓰레드에 뿌려서 parallel simulation 을 수행해야 한다. 하지만 이러한 synchronous 방식은 동기화 과정에서의 오버헤드가 크기 때문에 이를 최소화하기 위해 asynchronous 방식을 사용하는데, 각 액터에서 그라디언트를 서버로 전송하면 서버는 여러 액터들의 시뮬레이션 종료까지 기다려서 동기화를 하는 대신 바로 네트워크를 업데이트하고 이 네트워크를 액터로 다시 내려보낸다. 즉, 액터-마스터 간 동기화 과정은 있지만 액터끼리는 동기화가 되지 않는 것이다. 

따라서 각 액터들은 서로다른 네트워크를 사용하여 시뮬레이션을 진행하게 되며, global network 는 과거의 파라메터가 만든 그라디언트로 업데이트가 진행되게 된다. 예를 들어 설명해보자. 4개의 워커가 있을 때, global network $\theta_1$ 로 처음 시뮬레이션을 시작한다고 하고, 시뮬레이션은 워커 순서대로 종료되었다고 가정하자. 그러면 이 때 워커1 의 시뮬레이션이 끝나고 그라디언트 업데이트를 하게 되면 global network 가 $\theta_2$ 로 업데이트되는데, 이후 워커2 의 시뮬레이션이 종료되면 $\theta_1$ 에 의해 생성된 그라디언트들이 $\theta_2$ 를 업데이트 하게 된다! 이는 워커4 로 가면 더 심해져서 $\theta_1$ 에 의해 생성된 그라디언트로 $\theta_4$ 를 업데이트 하게 될 것이다.

이러한 방식의 네트워크 트레이닝을 asynchronous training 이라고 하며, 과거 노드 간 커뮤니케이션 코스트가 심하던 시절 분산 트레이닝에도 활용되어 성과를 보였다. RL 에서도 예전부터 이러한 방식의 optimization 이 연구되었으며 잘 작동한다고 알려져 있다.

이 논문에서는 A3C 만 제안한 것은 아니고 asynchronous DQN 등 여러 알고리즘에 대해서 적용해 보았으나 A3C 가 가장 성능이 좋았다. A3C 에는 parallel training 외에도 몇 가지 트릭이 더 쓰였는데, 1) entropy regularization 을 통해 exploration 을 강화하고 2) 각 워커마다 exploration policy 를 다르게 주어 보다 다양한 경험을 수집하고 3) multi-step learning (n-step TD) 을 통해 variance 를 잡았고 4) 네트워크에 LSTM 을 붙여 sequential modeling 까지 잡았다.

## GAE

Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).

- Key idea: TD(λ) 의 advantage function version

앞서 언급했듯 RL 의 주요한 이슈중 하나는 sampling 으로부터 생기는 variance 를 어떻게 잡느냐다. 이 문제는 오래전부터 연구되어 왔는데, 먼저 1-step TD 를 n-step TD 로 변경하여 variance 를 잡는 방법이 있다. 앞서도 설명했지만, n-step TD 는 n-step return 을 target 으로 사용한다:

$$
G^{(n)}_t = R_{t+1}+\gamma R_{t+2}+...+\gamma^{n-1}R_{t+n}+\gamma^n V(S_{t+n})
$$

이 방법은 위의 A3C 나 Deep Q-learning 계열의 Rainbow 에서도 사용된 방법이나, n 이라는 하이퍼파라메터가 새로이 추가된다는 문제가 있다. 이러한 문제에 대응하여 나온 방법이 여러 n-step TD 를 averaging 하여 같이 사용하는 TD(λ) 다. 이를 위한 λ-return은 각 n-step return 을 averaging 한다:

$$
G^\lambda_t=(1-\lambda)\sum^\infty_{n=1}\lambda^{n-1}G^{(n)}_t
$$

Advantage function 이 variance reduction 에 도움이 된다는 것은 잘 알려져 있다. GAE 에서는 이를 활용하여 advantage function 의 λ-return 을 유도한다.

$$
\begin{align}
&\hat A^{\text{GAE}(\gamma,\lambda)}_t = \sum^\infty_{l=0} (\gamma \lambda)^l \delta^V_{t+l} \\
&\delta^V_t=r_t+\gamma V(s_{t+1})-V(s_t)
\end{align}
$$

## NPG

Kakade, Sham M. "A natural policy gradient." Advances in neural information processing systems. 2002.

Natural policy gradients (NPG) 를 이해하려면 먼저 natural gradients 에 대해 이해해야 한다. 

## TRPO

Schulman, John, et al. "Trust Region Policy Optimization." Icml. Vol. 37. 2015.

## PPO

Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
