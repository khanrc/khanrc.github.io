---
layout: post
title: "RL - (WIP) Transfer and Multitask RL"
tags: ['RL']
date: 2019-03-30
comments: true
---

* TOC
{:toc}

# Transfer and Multitask RL

Supervised Learning (SL) 에서의 transfer learning 이나 multitask learning 처럼, RL 에서도 같은 개념을 적용할 수 있다. 어떤 환경에서 학습한 policy/value, feature extractor 등을 다른 환경에 가져다 활용하는 transfer RL 이라던가, 각종 Atari 2600 game 들에서 general 하게 잘 작동하는 multitask RL 도 가능하며, 새로운 task 를 만났을 때 빠르게 학습할 수 있도록 도와주는 meta-learner 를 도입하여 meta RL 까지도 확장된다.

<!-- ## UVFA (Universal Value Function Approximator)

SKIP -->

## HER

Andrychowicz, Marcin, et al. "Hindsight experience replay." Advances in Neural Information Processing Systems. 2017.

- Key intuition: 사람처럼 RL agent 도 실패로부터도 배우자.
- Key idea: 실제로 goal 을 달성하지 못했더라도 reward 를 주어, 적극적으로 움직이는 agent 를 만들면 결국 실제 goal 을 달성할 수 있다.

> HER 은 사실 sparse reward problem 을 풀기 위해 multitask method 를 활용하는 방법이지 transfer 혹은 multitask problem 을 풀기 위해 제안된 방법이 아니다. 다만 다른 카테고리에 넣기에도 애매한 부분이 있어서 spinning up 에서 분류한 그대로 transfer and multitask RL 카테고리에 넣었다. 개인적으로는 exploration 방법론에 가깝다고 생각한다.

Hindsight Experience Replay (HER) 은 매우 재미있는 방법론이다. 사람과 RL의 학습 방법 중 주요한 차이 중 하나는, 사람은 실패로부터도 배우지만 RL agent 는 성공으로부터만 배운다는 것이다. 하키에서 공을 골대에 넣는 연습을 한다고 생각해보자. 공을 쳤는데 파워는 훌륭하지만 방향이 잘못되었을 경우, 사람은 방향만 잘 조절하면 되겠다고 생각하지만, agent 에게는 그저 실패인 경험으로 남는다. 

RL 에서도 이렇게 실패로부터도 배우게 하고 싶다면 domain knowledge 를 통해 reward 를 가공하는 방법이 있다. 공의 파워와 방향을 분리하여 둘중 하나가 맞았다면 골대에 들어가지 않았더라도 약간의 reward 를 주는 방식이다. 이를 reward engineering 혹은 reward shaping 이라고 부르며, sparse reward 를 dense reward 로 바꿔준다. 문제는 이 작업이 매우 domain specific 하고 까다로운 작업이기 때문에 이 과정을 없애는 것이 RL 의 주요한 챌린지가 되며, 그렇기 때문에 지금까지 앞서 포스트들에서도 sparse reward problem 을 다루는 알고리즘들을 살펴보았다.

HER 에서도 마찬가지로 이러한 sparse reward problem 을 다루는데, 지난한 reward engineering 없이도 실패로부터도 배우도록 한다. 마치 마법같은 이야기이지만 방법은 매우 간단하다. Agent 가 timestep T 이후에 goal $g$ 에 도달하지 못했더라도, final state $s_T$ 가 마치 goal 인 것처럼 reward 를 준다. 이렇게 되면 agent 는 goal 로 가는 방법은 아니더라도 $s_T$ 로 가는 방법은 배우게 되므로, 이후에 random policy 에 비해 더 많은 곳을 가볼 수 있게 되어, 최종적으로 언젠가 goal 에 도달할 수 있게 해 준다.

![her-algorithm](/assets/rl/tm-her-algo.png){:.center width="70%"}
*HER pseudo code*

HER 은 experience replay 방법론이므로 각종 off-policy method 와 함께 사용할 수 있다. 또한 모든 final state 를 goal 로 취급하면 실제 goal 과 구분이 안 되므로, 일부 final state 만 랜덤하게 sampling 하여 goal 로 취급하고, goal 로 취급하기 전의 original trajectory 도 함께 experience replay 에 넣음으로써 실제 goal 의 경우 보다 높은 리워드를 받을 수 있게 해 준다.



<!-- Intrinsic motivation 파트에서는 sparse reward problem 을 풀기 위해 uncertainty 기반의 exploration 을 제안하였다. 하지만 search space 가 정말 크고 매우 sparse reward 인 문제를 생각해보자. Uncertainty 기반 exploration 을 한다고 하더라도, 시작 위치 근처를 전부 탐색했는데 여전히 reward 를 받지 못했다면 문제가 발생한다. Agent 의 입장에서 시작 위치 근방은 전부 certain 한 state 이므로 더이상 uncertainty 기반 exploration 은 도움이 되지 않으며, 동시에 여전히 policy 는 reward 를 받지 못했으므로 random policy 에서 벗어나지 못했다. Random policy 로 certain 영역 밖으로 벗어나야만 uncertainty 기반 exploration 이 작동할 텐데,  -->


## PathNet

Fernando, Chrisantha, et al. "Pathnet: Evolution channels gradient descent in super neural networks." arXiv preprint arXiv:1701.08734 (2017).

