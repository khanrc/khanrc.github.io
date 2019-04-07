---
layout: post
title: "RL - (WIP) Model-based RL"
tags: ['RL']
date: 2019-04-07
comments: true
---

* TOC
{:toc}

# Model-based RL



## I2A (Imagination-Augmented Agents)

Racanière, Sébastien, et al. "Imagination-augmented agents for deep reinforcement learning." Advances in neural information processing systems. 2017.

## World models

Ha, David, and Jürgen Schmidhuber. "Recurrent world models facilitate policy evolution." Advances in Neural Information Processing Systems. 2018.



## AGZ / AZ / EXIT

- AlphaGo Zero (AGZ): Silver, David, et al. "Mastering the game of go without human knowledge." Nature 550.7676 (2017): 354.
- AlphaZero (AZ): Silver, David, et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." Science 362.6419 (2018): 1140-1144.
- Expert Iteration (EXIT): Anthony, Thomas, Zheng Tian, and David Barber. "Thinking fast and slow with deep learning and tree search." Advances in Neural Information Processing Systems. 2017.

- Key idea: 모델을 알면 planning 이 가능하다. Planning 을 통해 현재 policy 보다 더 좋은 planned policy 를 사용하고, 이를 타겟으로 학습하여 current policy 를 업데이트하자.

이 세 논문은 위 두 논문과 조금 다른데, 모델이 주어진 경우를 다룬다. 보통의 RL problem 은 state-action pair (s,a) 가 결정되면 environment 와의 interaction 을 통해 s' 을 알아내야 하지만, 모델이 주어지면 environment 와의 interaction 없이도 다음 state s' 을 알 수 있다.

사실 더 정확히 말하면 모델이 주어졌다고 말하기 어려운 부분이 있다. 플레이어 입장에서 다음 상태 s' 은 상대 플레이 결과까지 포함되어야 한다. 체스로 예를 들면, 말을 옮기는 액션을 했을 때 말이 어떻게 이동할지는 알 수 있지만, 실제로 내가 다시 decision making 을 해야 하는 상태는 그 상태가 아니라 상대 플레이어도 말을 옮긴 뒤다. 따라서 상대 플레이 결과까지 감안한다면 사실 model is given 케이스가 아닌 것이다. 

하지만 대전게임의 특성상 상대 플레이를 내가 대신 플레이함으로써 채울 수 있다. Exact model 까지는 아니더라도 approximated model 정도는 알 수 있는 것이다. 이러한 특성에 따라 1:1 대전게임에서는 상대의 policy 를 내 policy 로 대체함으로써 model 을 완성하고, 이를 통해 planning 을 하여 현재의 current policy 보다 더 좋은 decision making 을 하는 방식이 사용된다. 이 때 planning algorithm 으로 사용되는 것이 바로 Monte Carlo Tree Search (MCTS) 다.

![mcts](/assets/rl/mb-mcts.png){:.center}
*MCTS in AGZ*

이와 같이 planning 을 통해 현재 policy 보다 더 좋은 planned policy 를 얻었다면, 이 planned policy 를 target 으로 하여 학습할 수 있다. 또한 MCTS 에서는 policy function P(a|s) 뿐만 아니라 현재 state 를 평가하는 value function V(s) 도 필요한데, 이는 게임이 끝난 후 승패를 reward 로 하여 학습한다.

### AlphaGo for everybody

위에서는 지금까지의 흐름에 맞추어 설명했다면, 여기서는 그냥 최대한 직관적으로 알파고의 알고리즘에 대해 설명한다. 사실 알파고의 플레이 방식과 학습 방식은 사람의 그것과 매우 닮아있다.

![mb-shogi](/assets/rl/mb-shogi.jpg){:.center}

우리가 장기를 둘 때를 생각해보자. 먼저 장기판을 보고 움직일 만한 수를 생각한다. 위 상황에서 한 (빨간색 말) 의 차례라면, 왼쪽 아래의 마를 움직여 상을 잡는다거나, 제일 오른쪽 쫄을 옆으로 옮겨 오른쪽 마가 나갈 길을 만든다거나 여러 후보 수가 있을 것이다. 중요한 것은, 사람은 경험적 직관에 의해 현재 장기판으로부터 몇가지 후보수를 추려내어 생각하지 모든 경우의 수를 다 따지지 않는다.

후보수를 추려냈다면 그 다음 스텝은 수읽기다. 내가 이렇게 두었을 때 상대가 어떻게 둘지, 그러면 나는 다시 어떻게 둘지를 고민하여 머릿속으로 여러 시뮬레이션을 수행하고, 그 결과 처음에는 좋아 보였던 수를 버리기도 하고 처음에는 크게 생각하지 않았던 수를 결국 고르기도 한다.

이 수읽기 과정에서 현재 상태의 판세를 읽는 직관이 필요하다. 머릿속으로 몇 수 플레이 해본 다음 어떤 상황이 될지를 놓고 그 상황이 유리한지 불리한지 판단할 수 있어야 한다. 이 능력이 부족하면 수읽기를 열심히 해도 별다른 도움이 안 될 것이다.

사람이 장기를 둘 때 필요한 능력은 이렇듯 수읽기와 두가지 직관이다. 알파고는 이 세가지 요소를 그대로 가져가는데, 현재 상태에서 후보 수를 결정하는 policy network, 이를 기반으로 수를 읽는 MCTS, 수읽기를 할 때 현재 판세를 판단하는 value network 로 구성된다.

학습 과정도 사람과 거의 유사하다. 사람이 장기 공부를 한다면, 아마 먼저 책을 보고 학원을 다녀서 다른 사람들의 플레이를 모방하고 따라하며 실력을 키울 것이다. 그리고 다른 사람과 대전하면서 점차 경험을 쌓아 실력을 증진시킬 것이다. 이 경험이 쌓인다는 것은 여러 요소가 있을 것인데, 먼저 수읽기를 해서 고른 최종 수가 잘 먹혀 경기에 이기면 다음번에는 유사한 상황이 되었을 때 그만큼의 수읽기를 하지 않더라도 이전의 경험을 살려 같은 플레이를 할 수 있을 것이다. 반대로 