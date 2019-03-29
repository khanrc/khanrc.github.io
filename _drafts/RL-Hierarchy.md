---
layout: post
title: "RL - Hierarchical RL"
tags: ['RL']
date: 2019-03-24
comments: true
---

* TOC
{:toc}


# Hierarchical RL 

RL 의 agent 는 micro action 으로 움직인다. Intrinsic motivation 예제에서 보았던 몬테주마의 복수 게임을 다시 생각해보자.

![montezuma](/assets/rl/hrl-montezuma.png){:width="60%" .center}

게임을 해본 적은 없지만, 아마 이 게임에서 액션이라고 하면 좌우 이동과 점프 그리고 사다리가 있다면 내려가는 4개의 상하좌우 키로 구성되어 있을 것이다. 일반적으로 RL 에이전트는 이 액션을 그대로 가져가서, 각 state 에서 상하좌우 중 하나의 액션을 취하고, environment 가 다음 state 를 돌려주면 다시 그에 맞는 액션을 취하는 방식으로 동작하게 된다.

하지만 사람은 이렇게 행동하지 않는다. Intrinsic motivation 에서 설명했듯 몬테주마의 복수 게임은 먼저 열쇠를 먹고 문으로 가야 하는 게임이므로, (충분히 학습된) 사람이 게임을 하면 먼저 1) 열쇠를 먹고 2) 문으로 갈 것이다. 열쇠를 먹으러 갈때 장애물들을 피해서 갈 것이고, 장애물들을 피해 열쇠로 향하기 위해 각 state 마다 적절한 action 을 취할 것이다. 즉 사람은 액션을 정함에 있어서 hierarchy 가 있어서, 먼저 상위 액션을 결정하고, 상위 액션을 취하기 위해 하위 액션들을 결정하게 된다. Hierarchical RL (HRL) 은 이러한 관점을 따르는 접근법이다.

## FuN

Vezhnevets, Alexander Sasha, et al. "Feudal networks for hierarchical reinforcement learning." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.

FuN 에서는 hierarchy 를 주기 위해 micro action 을 수행하는 worker 와, worker 의 goal 을 만들어주는 manager 가 존재한다. Manager 는 상위 액션을 결정하여 worker 에게 goal 을 부여하고, worker 는 이 goal 을 달성하기 위해 움직이게 된다.

![fun-arch](/assets/rl/hrl-fun-arch.png){:.center}
*Architecture of FuN algorithm*

Manager 는 이를 위해 state 를 받아서 goal embedding 을 생성하고, worker 는 먼저 action embedding 을 생성한 뒤 goal embedding 과의 연산을 통해서 최종 action 을 결정하게 된다.

이 구조는 사실 fully differentiable 하기 때문에 worker 의 policy gradient 를 그대로 흘려넣어서 학습이 가능하다. 하지만 이렇게 되면 manager 가 high-level action 을 잘 생성하도록 학습되리라는 보장이 없기 때문에 의도적으로 중간에 gradient flow 를 끊어주고 transition policy gradient 라는 방법으로 학습한다. $A_t^M$ 이 manager 의 advantage function 이고 $g_t$ 가 goal embedding 이라고 할 때, transition policy gradient $\nabla g_t$ 는:

$$
\nabla g_t = A_t^M \nabla_\theta d_{cos}(s_{t+c}-s_t, g_t(\theta)) \\
\text{where} \, A_t^M=R_t-V_t^M(x_t, \theta)
$$

가 된다. 여기서 $d_{cos}$ 는 cosine similarity 이고, s 는 manager 의 state embedding, x 가 original observation 이다. c step 동안 worker 가 goal 방향으로 잘 이동했는지와 그랬을 때의 advantage value 가 어떻게 되는지를 고려하여 loss 를 계산한다. 

위 수식은 policy gradient 수식과 동일한 직관적 해석을 갖는다. Current policy 가 좋은 결과를 얻은 경우 (positive advantage value) 해당 trajectory 대로 움직이도록 action 을 업데이트하고, 좋지 못한 결과를 얻었을 경우 (negative advatnage value) 해당 trajectory 와 다르게 행동하도록 action 을 업데이트한다. 여기서 manager 의 action 이 곧 goal $g_t$ 다.

Worker 는 manager 가 생성한 goal embedding 방향으로 잘 이동하였는지에 따라 intrinsic reward 를 받는다:

$$
r_t^I = \frac 1c \sum^C_{i=1} d_{cos}(s_t-s_{t-i}, g_{t-i})
$$

최종적으로 worker 가 받는 reward $R_t^W$ 는 intrinsic reward $R_t^I$ 와 environment reward $R_t$ 를 섞는다.

$$
R_t^W = R_t + \alpha R_t^I
$$

## HIRO

Nachum, Ofir, et al. "Data-efficient hierarchical reinforcement learning." Advances in Neural Information Processing Systems. 2018.

- Key idea: simplified FuN + TD3 + off-policy correction

FuN 은 다소 복잡한 부분이 있다. HIRO 에서는 manager 를 high-level policy, worker 를 low-level policy 라 하며, high-level policy 가 goal embedding 을 생성하는 것이 아니라 goal state 자체를 생성한다. Low-level policy 는 따로 goal embedding 을 입력받지 않고 intrinsic reward 를 계산할 때에만 goal 을 이용하여 계산한다. FuN 에서는 goal 의 direction 만 고려하는 것이 성능 향상에 도움이 되었다고 이야기하고 있으나 HIRO 에서는 position 을 사용한다.

![hiro-arch](/assets/rl/hrl-hiro-arch.png){:.center}
*Architecture and algorithm of HIRO*

HIRO 의 구조와 알고리즘은 위 그림 한장에 잘 설명되어 있다. High-level policy 는 c step 마다 goal state 를 생성하고, 이것이 곧 high-level policy 의 action 이다. c step 동안 얻은 environment reward 를 모아서 high-level transition $(s_t, \tilde g_t, \sum R_{t:t+c-1}, s_{t+c})$ 를 얻을 수 있다. FuN 과 같이 high-level policy 의 action 이 곧 goal 이므로, 위 transition 은 (s, a, r, s') 에 해당한다.

Low-level policy 를 위한 intrinsic reward 는 goal state 와의 distance 로 계산된다. Goal state 를 현재 state $s_t$ 에서의 relative position 으로 다루기 때문에 state 가 바뀌면 goal state 도 따라 바뀌게 되고, 이를 위해 goal transition function h 를 사용한다:

$$
h(s_t,g_t,s_{t+1})=s_t+g_t-s_{t+1}
$$

이제 각 타임스텝의 goal state 를 알았으니 이를 이용하여 distance 를 계산하여 reward 로 사용한다:

$$
r(s_t, g_t, a_t, s_{t+1})= - \lVert s_t + g_t - s_{t+1} \rVert
$$

이제 각 policy 의 reward 를 계산하였으니 TD3 으로 학습을 하면 되는데, 위 high-level transition 을 잘 보면 goal 에 tilde 가 붙어있는 것을 확인할 수 있다. TD3 과 다르게 HIRO 에서는 low-level policy 가 존재하므로 같은 (s, g) 라 하더라도 다른 low-level policy 를 갖는다면 그 trajectory 가 달라진다. 따라서 학습할 때 current low-level policy 에 맞게 goal state 를 조정해주는 off-policy correction 작업이 필요하다. Current low-level policy 를 따를 때 해당 trajectory 가 나오도록 goal state 를 조정한다:

$$
\arg\min_{g_t} \sum^{t+c-1}_{i=t} \lVert a_i - \mu^{lo}(s_i, \tilde g_i) \rVert^2_2
$$

실제로 구현할 때에는 $\tilde g_t$ 를 적당히 몇개 샘플링해서 위 값을 측정해보고 제일 적게 나오는 goal 을 사용한다.


## STRAW

Vezhnevets, Alexander, et al. "Strategic attentive writer for learning macro-actions." Advances in neural information processing systems. 2016.
