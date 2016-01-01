---
layout: post
title: "Restricted Boltzmann Machine"
tags: ['DataScience/Deep Learning']
date: 2015-07-14 15:23:00
---
# Restricted Boltzmann Machine

## Boltzmann Machine

먼저, **Boltzmann Machine**부터 살펴보자:

![BM](http://cfile7.uf.tistory.com/image/2152B63D53DCCE511F64E2)

이러한 형태로, 모든 노드가 전부 연결되어 있는 것을 Boltzmann Machine이라고 한다. 이 때 각 노드들은 확률적으로 정의되고 이 확률이 Boltzmann distribution을 따르기 때문에 Boltzmann Machine이다. 이 형태는 수학적으로 stochastic recurrent neural network라고 할 수 있는데, 각 노드들이 확률적으로 정의되고 모든 노드들이 연결되어 순환하므로 recurrent라 한다. 그래프에서 화살표가 없는데, 이 그래프는 undirected graph다.

위에서 v는 visible node로, 우리가 볼 수 있는 노드를 의미한다. 이는 input node와 비슷한 개념으로 이해할 수 있다. input data는 우리가 관찰할 수 있는 데이터고, 즉 visible node가 된다. 반면 h는 hidden node로, 우리가 볼 수 없는 노드다. 이 모든 것은 확률로 정의되는데, 이는 아래 Boltzmann distribution에 의한다:

**Boltzmann distribution:**   


여기서 p는 확률을 나타내고, s는 노드들의 상태(state), E는 에너지, k는 볼츠만 상수, T는 온도를 의미한다. 역학 모델을 차용한 것이지 실제 물리적 상황이 아니므로, kT는 계산의 편의를 위해 1로 놓는다. 또한 확률분포에서 확률의 총합은 1이 되어야 하므로 normalizing factor Z를 적용하면 위 식은 아래와 같다.   


결국 중요한 것은 에너지다. 에너지는 낮을 수록 안정적이고, 이 모델은 그래프가 가장 안정적인 경우를 가장 가능성(probable) 있는 상태로 판단한다(위 식에서 에너지가 낮을 수록 확률이 높다). 그럼 이 때 에너지 E의 구조를 보자.

**Energy of Boltzmann machine:**   


는 i번째 노드의 값이고, 0 또는 1이다. 는 노드의 bias다. 는 노드i와 j를 잇는 엣지의 weight다.   
즉, 활성화(activate)된 모든 노드의 bias와 활성화된 노드간의 엣지 weight를 전부 더해서 음수를 취하면 에너지 E를 구할 수 있다. 

## Restricted Boltzmann Machine

자 그런데 여기서 문제는, 이렇게 전부 서로 연결되어 있는 BM은 그 dependency 때문에 학습이 매우 어렵다. 때문에 이에 몇가지 제한을 걸어 학습이 쉽도록 한 것이 바로 **Restricted Boltzmann Machine**, RBM이다:

![RBM](http://cfile10.uf.tistory.com/image/24178B3953DCD4A7135C6E)

그림에서 볼 수 있다시피 visible units간, hidden units간 엣지가 없어졌다. 즉, 같은 레이어 내부에서는 커넥션을 없앤 것이다. 확률적으로 말하자면, 각 레이어의 joint probability를 없애고 레이어 내부의 변수들(variables)이 서로 독립적(independent)이라고 가정한다.

이제, 이에 따른 에너지와 확률은 다음과 같다.

**Energy and Probability of a Restricted Boltzmann Machine:**   


a와 b는 각각 visible units과 hidden units의 bias이고, v와 h는 상태를 나타낸다. 이 때 확률은 Boltzmann distribution에 따라   


이다. 에너지가 낮을 수록 확률이 높아진다.

## Training

RBM은 unsupervised feature learning이고, 이 모델의 학습은 트레이닝 데이터들의 분포를 나타내는 가장 안정적인 에너지 모델 E를 찾는 것이다. 즉, 트레이닝 데이터 에서의 확률분포 을 최대화하는 것이라고 할 수 있다. 따라서 이에 대한 log-likelihood 학습이 가능하다. 이하 수식들은 [deepcumen - RBM](http://deepcumen.com/tag/rbm/)에 자세히 나와 있으니 이해가 안 가는 부분은 해당 문서를 참고하자.

여기서 이다. 이로부터, log-likelihood의 최대값을 구해야 하므로 Gradient Ascent를 사용한다. 따라서 위 식을 미분하면   


이 된다. 이 식에서 첫번째 항은 주어진 에 대해 모든 에 대한 의 기대값이므로 input data에 대한 값이다. 반면 두번째 항은 모든 에 대한 의 기대값이므로 input data가 아니라 전체 model에 대한 값이다. 따라서 이는 아래와 같이 요약할 수 있다:

은 기대값을 의미한다. 이제, 이 식을 MCMC Gibbs Sampler Contrastive Divergence를 사용해서 학습할 수 있다. 위 식에서 input data에 대한 첫번째 항은 쉽게 구할 수 있지만 모델 전체에 대한 두번째 항은 구하기가 쉽지 않다. 이를 Gibbs Sampling을 이용한 MCMC 추정을 사용해서, 모든 에 대해 계산하지 않고 와 를 통해 를 샘플링하여 근사할 수 있다.

그러나 여전히 MCMC의 수렴조건을 계산하는 것이 까다로운데, 이를 (Contrastive Divergence - k) 를 통해 위 근사값을 다시 한번 근사한다. 최종적인 log-likelihood 기울기(미분값)는 아래와 같다:

이 때 k의 값은 1로 하여도 큰 문제가 없다고 알려져 있다. 지금까지의 과정에 파라메터 업데이트까지 포함하여 아래와 같이 정리할 수 있다.

![RBM training process](http://deepcumen.com/wp-content/uploads/2015/04/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7-2015-04-24-%EC%98%A4%EC%A0%84-3.50.29.png)

### Conclusion

결과적으로, RBM은 hidden units로부터 visible units를 재구성할 수 있고, 이는 autoencoder의 hidden layer의 역할과 동일하다. 즉, 이렇게 학습된 RBM의 hidden units는 각각 압축된(compressed) feature가 되므로 RBM은 unsupervised feature learning이라 한다.

## References

[deepcumen - RBM](http://deepcumen.com/tag/rbm/): 이전 챕터인 EBM부터 보자.   
[Deep Learning이란 무엇일까?](http://enginius.tistory.com/499): RBM에 대해서 쉽게 살펴볼 수 있다.   
[[Deep Learning ]RBM trained with Contrastive Divergence](http://enginius.tistory.com/315): RBM의 학습법(Contrastive Divergence)에 대해서 다룬다.

[DeeplearningNet; RBM](http://deeplearning.net/tutorial/rbm.html)   
[Wiki; RBM](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)

최근에 느끼는 건데, 참고 아티클들은 전부 돌아가면서 읽으면 서로서로 보완재로 작용해서 이해하는 데 도움이 된다.


[Tistory 원문보기](http://khanrc.tistory.com/106)
