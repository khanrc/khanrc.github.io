---
layout: post
title: "RL - (WIP) Model-based RL"
tags: ['RL']
date: 2019-04-15
comments: true
---

* TOC
{:toc}

# Model-based RL

Environment 를 모델링 한 것을 model 이라고 하며, 이를 활용하는 경우를 model-based RL 이라고 한다. 처음부터 모델을 알고 있는 케이스와 모델을 학습해서 사용하는 케이스로 나뉜다. 

## Model is Given

보통의 RL problem 은 state-action pair (s,a) 가 결정되면 environment 와의 interaction 을 통해 s' 을 알아내야 하지만, 모델이 주어지면 environment 와의 interaction 없이도 다음 state s' 을 알 수 있다. 그렇게 되면 이를 기반으로 planning 을 할 수 있고, 기존의 model-free RL 에서는 할 수 없던 방식의 접근이 가능해진다.

### AGZ / AZ / EXIT

AlphaGo Zero (AGZ): Silver, David, et al. "Mastering the game of go without human knowledge." Nature 550.7676 (2017): 354.

AlphaZero (AZ): Silver, David, et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." Science 362.6419 (2018): 1140-1144.

Expert Iteration (EXIT): Anthony, Thomas, Zheng Tian, and David Barber. "Thinking fast and slow with deep learning and tree search." Advances in Neural Information Processing Systems. 2017.

- Key idea: Planning 을 통해 현재 policy 보다 더 좋은 planned policy 를 사용하고, 이를 타겟으로 학습하여 current policy 를 업데이트하자.

위 세 논문은 1:1 턴제 보드게임을 다룬다. 각 논문마다 바둑, 체스, 장기, Hex 등 서로 다루는 게임은 다르지만 결국 1:1 턴제 보드게임으로 거의 동일한 알고리즘을 사용한다. Model is given 케이스로 설명했지만, 사실 정확히 말하면 이 문제에서 모델이 주어졌다고 말하기 어려운 부분이 있다. 플레이어 입장에서 다음 상태 s' 은 상대방의 플레이 결과까지 포함되어야 한다. 체스로 예를 들면, 말을 옮기는 액션을 했을 때 말이 어떻게 이동할지는 알 수 있지만, 실제로 내가 다시 decision making 을 해야 하는 상태는 그 상태가 아니라 상대 플레이어도 말을 옮긴 뒤다. 하지만 상대 플레이어가 어떻게 말을 옮길지 모르기 때문에, 이 부분까지 감안한다면 사실 model is given 케이스가 아닌 것이다. 

하지만 1:1 대전게임의 특성상 상대 플레이를 내가 대신 플레이함으로써 채울 수 있다. Exact model 까지는 아니더라도 approximated model 정도는 알 수 있는 것이다. 이러한 특성에 따라 1:1 대전게임에서는 상대의 policy 를 내 policy 로 대체함으로써 model 을 완성하고, 이를 통해 planning 을 하여 현재의 current policy 보다 더 좋은 decision making 을 하는 방식이 사용된다. 이 때 planning algorithm 으로 사용되는 것이 바로 Monte Carlo Tree Search (MCTS) 다.

![mcts](/assets/rl/mb-mcts.png){:.center}
*MCTS in AGZ*

이와 같이 planning 을 통해 현재 policy 보다 더 좋은 planned policy 를 얻었다면, 이 planned policy 를 target 으로 하여 학습할 수 있다. 또한 MCTS 에서는 policy function $P(a\|s)$ 뿐만 아니라 현재 state 를 평가하는 value function $V(s)$ 도 필요한데, 이는 게임이 끝난 후 승패를 reward 로 하여 학습한다. AGZ 에서는 MCTS visit counts 를 normalize 하여 target planned policy 를 생성하며, 이렇게 학습 시 oscillation 이나 catastrophic forgetting 현상이 없이 robust 하게 잘 학습됨을 보였다.

#### Self-play process in AZ/AGZ
{:.no_toc}

1. 현재 policy/value 를 사용하고, MCTS 에 exploration 을 위해 약간의 랜덤성을 가미하여 서로 붙인다. 서로 같은 모델이지만 랜덤성이 있기 때문에 어느정도는 서로 다른 플레이를 한다. 
2. 한 게임이 끝나면, 그 게임 결과를 사용해서 value network $V(s)$ 를 학습시키고, 매 수마다의 MCTS visit counts 를 normalize 하여 만드는 MCTS policy (target planned policy) 로 policy network $P(a\|s)$ 를 학습시킨다.
3. 업데이트되어 더 강해진 policy/value 을 사용하여 다시 1로 돌아가 self-play 를 반복한다.

사실 우리가 PG 논문들을 살펴보면서도 고민했듯 업데이트한다고 무조건 성능이 좋아지리란 보장이 있는 것은 아니다. 그래서 AGZ 에서는 성능 테스트를 진행하고 실제로 성능이 좋아졌는지를 체크하는 부분이 있다. 하지만 이후에 AZ 가 나오면서 그 부분이 빠졌고, 이는 매 스텝마다 성능이 좋아지리라는 보장은 없지만 계속 학습하면 어느정도 성능에 바운스가 있더라도 장기적으로는 계속 좋아진다는 것을 실험적으로 보인 셈이다.

## Model is Learned

위에서 살펴본 것처럼 모델을 알 수 있다면 참 좋겠지만, 사실 우리가 맞닥뜨리는 대부분의 문제는 그렇지 않다. 따라서 여기서는 모델도 같이 학습하여 활용하는 방식으로 접근한다.

### I2A (Imagination-Augmented Agents)

Racanière, Sébastien, et al. "Imagination-augmented agents for deep reinforcement learning." Advances in neural information processing systems. 2017.

- Key idea
  - Model-based RL + model-free RL
  - Planning with learned model

![I2a-arch](/assets/rl/mb-I2A-arch.png){:.center}

- Imagination core
  - Env model 은 pre-training 후 시작
    - model-free 학습 => 이 policy 로 env model 학습 => I2A 시작
  - Policy Net 은 I2A 의 imagination-augmented policy 를 따라하도록 학습
- Single imagination rollout
  - Rollout (simulation with imagination) 하고, 그 결과를 LSTM 을 태워서 rollout 정보를 인코딩
  - 이 롤아웃 정보는 이 state 에서 우리가 만든 model 로 rollout 을 해본 정보가 담겨 있음
- Full I2A architecture
  - A3C 사용
  - Model-free path
    - 그냥 current observation 으로부터 바로 policy / value 예측하는 모델
  - Imagination rollout 정보랑 model-free path 정보랑 싹 합쳐서 policy, value 만듦

### World models

Ha, David, and Jürgen Schmidhuber. "Recurrent world models facilitate policy evolution." Advances in Neural Information Processing Systems. 2018.

- Key idea: policy network 를 feature extractor 와 controller 로 분리시켜서 large network 을 사용할 수 있도록 함

보통 RL 에서 사용하는 policy network 는 image classification 등에서 사용하는 네트워크와 비교하면 훨씬 작은 모델을 사용한다. 이유는 - 잘 모르겠지만 아마도 학습이 빨리 되어야 하니까?

아무튼 그런데, 이 policy network 는 사실 environment 에 대한 이해를 담고 있는 feature extractor 와 그 feature 로부터 액션을 결정하는 controller 로 나누어 볼 수 있다. World model 에서는 environment 에 대해 먼저 model 을 학습시키고, 이 model 을 feature extractor 로 사용한다. 이렇게 되면 복잡한 environment 를 커다란 네트워크로 모델링이 가능해지고, 이 뒤에 간단한 controller 를 붙여 학습시킨다.

![world-model](/assets/rl/mb-worldmodel.png){:.center}

- V: Visual sensory component
- M: Predictive model
- C: Controller
- V + M = world model
  - 처음에 랜덤 폴리시로 학습
  - 랜덤 폴리시로 학습하면 초반밖에 못 보는 문제가 있음. 이 다음에 나온 Simple 논문에서는 그래서 폴리시 학습 이후 다시 world model 을 업데이트하는 방식을 제안
- C 는 CMA-ES (Covariance-Matrix Adaptation Evolutionary Strategy) 사용

I2A 는 모델을 만들어서 플래닝을 하여 성능을 개선시키겠다는 접근법이었다면, world model 은 policy 에서 feature extractor 를 분리시켜서 environment 를 잘 이해하도록 학습. 이후에 controller 를 학습시킴으로써 보다 좋은 policy 학습.