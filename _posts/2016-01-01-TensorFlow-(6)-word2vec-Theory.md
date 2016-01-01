---
layout: post
title:  "TensorFlow (6) word2vec - Theory"
categories: [design, tools]
tags: [tag1, tag2]
---


# [TensorFlow](http://www.tensorflow.org)
## [Vector Representations of Words](http://www.tensorflow.org/tutorials/word2vec/index.html)
word2vec or word embedding.

test $b_{t-1}$, $b_t$. $\dot{x}_1$

highlight ==test== !!  
quote "test" !!  
footnotes[^1].  
[^1]: testtest

### Highlights
* 왜 단어를 벡터로 나타내야 하는가?
* 모델의 개념과 어떻게 학습되는가
* 텐서플로를 통한 간단한 버전의 구현
* 간단한 버전을 좀 더 복잡하게

기본 버전인 [word2vec_basic.py](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/word2vec/word2vec_basic.py) 과 좀 더 진보된 버전인 [word2vec.py](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/embedding/word2vec.py) 를 제공하니 참고하자.

하지만 그 전에, 왜 word embedding을 해야 하는지를 먼저 살펴보자.  

### Motivation: Why Learn Word Embeddings?
이미지나 오디오 데이터는 dense데이터임. 얼굴인식이나 음성인식 등의 이미지/오디오 데이터를 사용하는 작업은 이 데이터 안에 모든 필요한 정보가 담겨있다 (사람의 경우를 생각해보자 - 이미지만 보고 그 이미지로부터 얼굴을 인식할 수 있다). 반면, NLP에서는 단어를 표현하기 위해 one-hot 벡터를 사용하고, 이 방법은 단어간의 관계를 나타내는데 아무 도움이 안 됨. one-hot 벡터 방식으로 데이터를 나타내는 것은 데이터의 sparsity 문제를 가질 뿐만 아니라 통계적인 모델을 학습할 때 굉장히 많은 데이터가 필요하게 된다 - 예를 들면 'cat'에 대해 학습한 내용은 'dog'에 대해 학습한 내용에 전혀 영향을 끼치지 못함. 실제로는 서로 충분히 유사한 개체임에도 불구하고. 이러한 문제를 해결하기 위해, 단어의 concept에 대한 feature를 벡터로 나타낸 것이 바로 word embedding이다.  

![audio images text](https://www.tensorflow.org/versions/master/images/audio-image-text.png)

[Vector space models](https://en.wikipedia.org/wiki/Vector_space_model) (VSMs) 는 연속적인 벡터 공간에서 단어를 나타낸다 (임베딩한다) - 비슷한 의미의 단어끼리 모이도록. VSM은 NLP에서 오래도록 다루어졌지만, 거의 모든 방법론들은 같은 컨텍스트의 단어들은 그 의미를 공유한다는 [Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis) 에 기반한다.

> Distributional Hypothesis는 언어학에서의 의미이론이다. 같은 컨텍스트에서 사용된 단어는 비슷한 의미를 가진다는. 즉 같이 사용된 주변 단어로부터 그 단어를 규정할 수 있다는 것.

이와 다른 접근법은 두가지 카테고리로 나눌 수 있다: count-based methods (e.g. [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)) 와 predictive methods ([neural probabilistic language models](http://www.scholarpedia.org/article/Neural_net_language_models)). Count-based method는 거대한 텍스트 코퍼스에서 단어들이 어떤 단어들과 같이 등장하는지를 세고, 이를 작고 dense한 벡터로 압축한다. Predictive model은 이미 학습된 주변 단어들로부터 타겟 단어의 벡터를 예측한다.

> **랭귀지 모델**은 자연어로 된 텍스트 (sequence of words) 에 대해, 단어 시퀀스의 중요한 통계적 분포 특징을 내포한다. 즉, 이 모델은 자연어 텍스트에 대해 통계적 단어 시퀀스 분포를 갖고, 이를 통해 특정 단어 시퀀스 뒤에 어떤 단어가 나올지 확률적 예측을 할 수 있다.
> **뉴럴 네트워크 랭귀지 모델**은 NN에 기반한 랭귀지 모델이다. curse of dimensionality 의 효과를 줄이는 distributed representation을 학습할 수 있다. 러닝 알고리즘의 컨텍스트에서 curse of dimensionality란, 복잡한 함수를 학습할 때 방대한 양의 트레이닝 데이터가 필요한 것을 가리킨다. 인풋 변수의 수가 증가하면 학습을 위한 데이터의 수도 지수적으로 증가한다. 

Word2vec은 계산-효율적인 (computationally-efficient) predictive model이다. word2vec은 CBOW (Continuous Bag-of-Words model) 와 Skip-Gram model 이라는 두가지 주요한 특징을 갖는다. 이 두 모델은 알고리즘적으로 유사한데, 단지 CBOW는 컨텍스트 (주변 단어들) 로부터 타겟 단어의 벡터를 예측하고 반대로 skip-gram 은 타겟 단어로부터 주변 컨텍스트 단어들의 벡터를 예측한다. 이 inversion은 이상해보일 수 있는데, 통계학적으로 CBOW는 전체 컨텍스트를 하나의 관찰로 다룸으로써 분산되어 있는 정보를 스무스하게 만드는 효과가 있다. 이는 작은 데이터셋에서 효과적이다. 반면 skip-gram은 모든 타겟 단어 - 컨텍스트 단어 페어 각각을 새로운 관찰로 다루고, 이는 커다란 데이터셋에서 효과적이다. 우리는 skip-gram 모델에 집중할것이다.

### Scaling up with Noise-Contrastive Training
#### [Maximum-Likelihood Estimation (MLE)](https://en.wikipedia.org/wiki/Maximum_likelihood)
들어가기 전에 MLE를 먼저 살펴보자. 이 개념이 매번 헷갈려서 정리도 한번 했었는데 아직도 헷갈림. -_-

MLE는 주어진 데이터의 통계모델의 파라메터를 추정하는 방법론이다. 예를 들어, 성장한 여성 펭귄의 키 분포를 알고 싶다고 하자. 모든 펭귄의 키를 다 잴수는 없다. 키 분포가 정규분포를 따른다고 가정하자. 그러면 이때 평균과 분산을 알면 전체 분포를 알 수 있다. 이를 어떻게 알 수 있을까? MLE는 여기서 일부 데이터를 기반으로 모집단의 파라메터, 즉 평균과 분산을 추정한다 - 측정한 데이터가 나올 확률이 가장 높은 모집단을 추정하는 방식으로.  

일반적으로, 고정된 수의 데이터와 통계 모델에 기반해서, MLE는 [likelihood function](https://en.wikipedia.org/wiki/Likelihood_function)을 최대화하는 파라메터를 선택한다. 직관적으로, 관찰된 데이터로부터 선택된 모델의 "agreement" 를 최대화한다. 그리고 결과 분포에서 주어진 데이터의 확률을 최대화한다. 

> likelihood function (or simply likelihood) : 통계모델의 파라메터의 함수. "probability" 와 비슷하게 쓰이지만, 통계학적으로 "결과" 냐 "파라메터" 냐의 차이가 있다. Probability는 정해진 파라메터를 기반으로 결과 함수를 설명할 때 쓰인다 - e.g. 동전을 10번 튀겼고, 공평한 동전이라면, 이 때 항상 앞면이 나올 probability는 얼마일까? 반면 likelihood는 주어진 결과를 기반으로 파라메터의 함수를 설명할때 쓰인다 - e.g. 동전을 10번 튀겼고, 10번 다 앞면이 나왔을 때, 이 동전이 공평할 likelihood는 얼마일까?

#### Return to Scaling up with Noise-Contrastive Training
Neural probabilistic language model은 이전 단어들 $h$ (for "history") 가 주어졌을 때 다음 단어 $w_t$ (for "target") 의 확률을 추정하는 MLE를 통해 학습된다. 이 과정은 softmax function에 기반한다:

$$
\begin{align}
P(w_t | h) &= \text{softmax}(\text{score}(w_t, h)) \\
           &= \frac{\exp \{ \text{score}(w_t, h) \} }
             {\sum_\text{Word w' in Vocab} \exp \{ \text{score}(w', h) \} }.
\end{align}
$$

여기서 $\text{score}(w_t, h)$ 는 타겟 단어 $w_t$ 와 컨텍스트 $h$ 의 공존 가능성 (compatibility) 를 계산한다 - 보통 dot product를 쓴다. 이 모델을 학습하기 위해 트레이닝 셋에 대해, log-likelihood를 최대화한다:

<div>
$$
\begin{align}
 J_\text{ML} &= \log P(w_t | h) \\
  &= \text{score}(w_t, h) -
     \log \left( \sum_\text{Word w' in Vocab} \exp \{ \text{score}(w', h) \} \right)
\end{align}
$$
</div>

> 근데 $P(w_t|h)$가 probability (posterior) 아닌가? likelihood면 $P(h|w_t)$ 여야 할 것 같은데...


이 방법은 적절하게 normalized된 probabilistic language model 을 학습하지만, 문제는 이 방법은 너무 비싸다. 다음에 어떤 단어가 나올지를 예측하기 위해 모든 단어들에 대해 확률을 전부 계산하고 노멀라이즈 해야 한다. 그리고 이 과정을 모든 training step마다 반복해야 한다.

![softmax-nplm](https://www.tensorflow.org/versions/master/images/softmax-nplm.png  "title" "width:60%")

반면, word2vec의 feature learning 에서는 full probabilistic model을 학습할 필요가 없다. CBOW와 skip-gram 모델이 binary classification object (logistic regression) 을 사용해서 학습하는 대신, 같은 컨텍스트에서 $k$개의 가상의 (noise) 단어 $\tilde w$로부터 타겟 단어 $w_t$ 를 구별한다. 아래는 CBOW에 대한 그림이다. skip-gram은 단지 방향만 반대로 하면 된다:

![CBOW](https://www.tensorflow.org/versions/master/images/nce-nplm.png  "title" "width:60%")

> 잘 이해는 안 가지만, 이미지를 참고하면, 원래는 모든 가능한 단어 $w'$에 대해 확률을 구해봐야 했지만 CBOW 혹은 skip-gram 에서는 k개의 imaginary (noise) 단어 $\tilde w$ 에 대해서만 테스트하여 학습 속도를 향상시킨다.

수학적으로, 이 예제에 대해, 다음 objective 를 최대화 하는 것을 목표로 한다:

<div>
$$
J_\text{NEG} = \log Q_\theta(D=1 |w_t, h) +
  k \mathop{\mathbb{E}}_{\tilde w \sim P_\text{noise}}
     \left[ \log Q_\theta(D = 0 |\tilde w, h) \right]
$$
</div>

$Q_\theta(D=1 | w, h)$ 는, 임베딩 벡터 $\theta$를 학습하면서, 데이터셋 $D$에서 컨텍스트 $h$하에서 단어 $w$가 나올 확률을 계산하는 binary logistic regression probability 모델이다. 실제 학습에서는, noise distribution으로부터 k contrastive words를 샘플링 (drawing) 함으로써 기대값 (expectation) 을 추정한다. (즉, [Monte Carlo average](https://en.wikipedia.org/wiki/Monte_Carlo_integration) 를 계산한다)

> Monte Carlo average (integration):  
> 몬테카를로 적분 (integration) 은 랜덤을 이용해서 적분하는 방법이다. 랜덤하게 난수를 발생시키고, 해당 난수가 적분 범위 안에 들어가는 확률을 계산하여 이를 통해 적분한다. 도형의 넓이를 계산한다고 생각해 보면 쉽게 이해할 수 있음.

> 여기서 임베딩 벡터 $\theta$ 는 word embedding에서 바로 그 임베딩 벡터를 말함.

위의 목적 함수 (objective) 는 real word에 높은 확률을 할당하고 noise word에 낮은 확률을 할당했을 때 최대화된다. 기술적으로, 이는 [Negative Sampling](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) 이라 불린다. 이 함수는 위 소프트맥스 함수 ($J_{ML}$) 를 근사하지만 훨씩 더 적은 계산량을 가지고, 이는 훨씬 빠른 학습속도를 제공한다. 우리는 정확하게는 이와 거의 유사한 [noise-contrastive estimation (NCE)](http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf) 를 사용한다. 이는 TensorFlow 에서 `tf.nn.nce_loss()`라는 함수로 제공하므로 편리하게 사용할 수 있다.


### The Skip-gram Model
`the quick brown fox jumped over the lazy dog` 

라는 데이터셋을 생각해보자. 'context' 라는 것은 다양하게 정의될 수 있지만, syntactic contexts는 보통 타겟 단어의 주변 단어를 가리킨다. 일단, 'context' 가 타겟 단어의 좌우 1칸을 가리킨다고 해 보자:

`([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...`

살펴보면 알겠지만 이는 `(context, target)` 쌍이다. skip-gram은 타겟 단어로부터 컨텍스트 단어를 예측한다. 즉, 우리는 'quick' 으로부터 'the' 와 'brown' 을 예측하고, 'brown' 으로부터 'quick' 과 'fox' 를 예측해야 한다. 자, 그러면 데이터셋을 `(input, output)` 으로 구성하면 이렇게 된다:

`(quick, the), (quick, brown), (brown, quick), (brown, fox), ...`

object function 은 데이터셋 전체에 대한 함수이지만, 우리는 학습을 위해 online 혹은 minibatch learning 을 사용한다. 이를 자세히 살펴보자. 보통 minibatch 에서 batch_size 는 16에서 512 사이다.

위 트레이닝 셋에서 제일 첫 번째 케이스로 트레이닝 스텝 $t$를 생각해보자. 우리의 목표는 `quick` 으로부터 `the` 를 예측하는 것이다. 먼저 noisy (contrastive) example 의 수를 나타내는 `num_noise` 를 선택해야 한다. noisy example 은 noise distribution 을 따르며, 이 분포는 일반적으로 unigram distribution $P(w)$ 이다. 간단하게 하기 위해 `num_noise=1` 이라 하고 noisy example 로는 `sheep` 을 사용하자. 그러면 loss function을 계산할 수 있다:
> unigram distribution $P(w)$ 라는 것은 전체 데이터셋에서 각 단어의 unigram으로 생성한 확률분포를 의미하는 듯. sheep 이 위 데이터셋에 없다는 것이 이상한데, 일단 위 예제는 데이터셋의 일부라고 생각해보자.

<div>
$
J^{(t)}_\text{NEG} = \log Q_\theta(D=1 | \text{the, quick}) +
  \log(Q_\theta(D=0 | \text{sheep, quick}))
$ 
</div>

이 과정의 목표는 임베딩 파라메터 $\theta$ 를 업데이트하여 object function 을 최적화 (여기서는 최대화) 하는 것이다. 이를 위해, 임베딩 파라메터 $\theta$ 에 대해 loss의 gradient를 계산한다. 여기서는 $\frac{\partial}{\partial \theta} J_\text{NEG}$ 를 계산한다 - TensorFlow는 이를 위한 함수를 제공한다. 이후 이 gradient의 방향으로 임베딩 파라메터를 약간 업데이트한다. 이 과정을 전체 데이터셋에 대해 반복하면, 임베딩 벡터는 점차 실제 단어의 위치로 이동한다 - real words와 noise words가 분리될때까지.

이 학습 과정을 [t-SNE dimensionality reduction technique](http://lvdmaaten.github.io/tsne/) 같은 방법을 사용해서 2차원 혹은 3차원 공간으로 차원축소하여 시각화 할 수 있다. 이 과정을 살펴보면, 우리가 원하는 대로 단어의 의미를 잘 추출하여 벡터공간에 임베딩하는 것을 확인할 수 있다:

![word2vec visualization](https://www.tensorflow.org/versions/master/images/linear-relationships.png)

즉, 이 벡터들은 기존의 NLP prediction task에서 훌륭한 특성으로 사용될 수 있다 - POS tagging or named entity recognition 등. [Collobert et al.](http://arxiv.org/pdf/1103.0398v1.pdf) 또는 [Turian et al.](http://www.aclweb.org/anthology/P10-1040) 을 참고하자.
