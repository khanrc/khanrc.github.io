---
layout: post
title: "Autoencoders"
---

최근의 딥러닝은 대부분 CNN과 RNN으로 귀결되지만, 이 두 모델은 제각각 데이터에 대한 제약이 있다. 정확하게 말하면, 데이터 도메인의 특성에 맞게 모델을 변형하여 퍼포먼스를 향상시켰다고 할 수 있다. 데이터가 공간적인 (spatial) 한 특성을 가지는 경우엔 CNN을, 시퀀셜한 특성을 가지는 경우엔 RNN을 사용한다. 그러나 데이터가 이 두가지에 해당하지 않는 경우에는 근본적인 MLP를 사용해야 하고, 이 MLP의 local minimum problem (이 문제를 이르는 말이 있었던 것 같은데 생각이 안 남) 을 해결하기 위한 unsupervised pretraining 방법론 중 가장 대표적이고 직관적인 것이 오토인코더다. 이 글에서는 [벤지오의 딥러닝 책](http://www.deeplearningbook.org/) 중 [14장 Autoencoders](http://www.deeplearningbook.org/contents/autoencoders.html) 챕터를 간단하게 정리한다. 

# Chap 14. Autoencoders
오토인코더는 인풋을 히든 레이어로 변환 (encoding) 시켰다가 다시 인풋을 재생성 (decoding - reconstruct) 하여 학습된다. 이렇게 학습한 인코더 (encoder) 와 디코더 (decoder) 는 인풋을 다른 공간으로 매핑한다는 면에서 커널 (kernel) 이라고 볼 수 있고, 히든 레이어의 activation value 들이 바로 인코딩된 **code** 에 해당한다. 이 코드는 다른 공간에 매핑된 값으로써 latent feature 라고 볼 수 있다.

기존의 오토인코더는 차원 축소 (dimensionality reduction) 혹은 특성 추출 (feature learning) 을 위해 사용되었지만, 최근에는 latent variable model과의 연결고리가 발견되면서 generative modeling을 위해 사용된다. 이 부분은 [챕터 20](http://www.deeplearningbook.org/contents/generative_models.html)에서 다룬다. 아마 Variational Autoencoders (20.10.3) 얘기인 듯.

## 14.1 Undercomplete Autoencoders
히든 유닛의 개수가 인풋 유닛보다 적은 걸 undercomplete 이라 한다. 이러한 언더컴플릿 오토인코더야말로 가장 직관적인 dimensionality reduction의 형태다. loss function $L(x,g(f(x)))$ 을 최소화하여 학습할 수 있다.

여기서 loss function이 MSE (mean squared error) 이고 디코더가 선형 (linear) 이면 이 undercomplete autoencoder는 PCA와 동일하다. 비선형 (non-linear) 인코더/디코더를 사용하고 오토인코더를 깊게 (deep) 쌓음으로써 PCA 보다 강력한 특성 추출기 (feature extractor) 를 학습할 수 있지만, 여느 NN이 그러하듯 과적합 (overfitting) 문제가 있다.

## 14.2 Regularized Autoencoders
마찬가지의 문제가 히든 유닛의 수와 인풋 유닛의 수가 동일할 때나 혹은 더 많을 때 (overcomplete) 에도 발생한다. 오버피팅 문제를 피하기 위해 네트워크를 얕게 (shallow) 만드는 등 네트워크의 capacity를 줄이는 방법도 있지만, regularized autoencoder는 네트워크의 capacity를 유지하면서도 인풋의 본질적인 특성을 학습한다.

여기서 다루는 regularized autoencoders 외에도, inference procedure를 갖추고 latent variable을 다루는 대부분의 generative model은 오토인코더의 일종으로 볼 수 있다. 이러한 generative modelling은 단순히 인풋을 아웃풋으로 카피하는 것을 학습하는 것이 아니라, 인풋의 확률모델을 학습하기 때문에 regularization 없이도 high-capacity overcomplete encoding 을 학습한다. 

### 14.2.1 Sparse Autoencoders
Sparse autoencoder는 오토인코더의 코드 레이어 $h$ 의 loss function에 간단한 sparsity penalty $\Omega(h)$ 를 추가한다:

$$L(x,g(f(x)))+\Omega(h)$$

이를 통해 오토인코더는 단순한 identity function (인풋을 아웃풋으로 그대로 되돌리는 함수) 을 학습하지 않고 본질적인 latent feature를 학습할 수 있다. Sparse regularizer는 MAP approximation 관점에서 직관적으로 이해할 수 있는 weight decay 등의 다른 regularizer와는 다르게, 베이지안 관점에서 직관적으로 이해하기 어렵다. Discriminant model에서 MAP는 $p(\theta|x)$ 를 최대화해야 하는데, log를 씌워 베이지안 룰을 통하면 이는 $\log{p(x|\theta)}+\log{p(\theta)}$ 와 같다. 여기서 $p(x|\theta)$ 는 데이터 likelihood 이고, $p(\theta)$ 는 파라메터의 prior 이다. 직관적으로 $\theta$ 의 크기가 작을수록 $p(\theta)$ 가 커진다 (파라메터의 크기가 작을수록 직관적으로 더 가능성이 높은 파라메터라고 볼 수 있다). 자세한 건 챕터 5.7을 참조하자. 반면 Regularized autoencoder는 데이터의 prior에 기반한 것이 아니라 데이터 자체에 기반한 것이기 때문에 이와 같은 맥락으로 해석할 수 없다.
> Weight decay는 loss function에 weight의 크기에 대한 term을 추가하여 학습하는 네트워크의 weight의 크기를 줄이는 방법이다.  
> 
> 위 논리는 discriminant model에 공통적으로 적용할 수 있다. 파라메터가 작을수록 가능성이 높다는 건 내 이해인데 확실히 맞는지 모르겠음.

Sparse autoencoder를 ML (maximum likelihood) 를 학습하는 generative model의 관점에서 생각해 보자. Visible variables $x$ 와 latent variable $h$ 에서 joint distribution $p_{model}(x,h)=p_{model}(h)p_{model}(x|h)$ 이다. 여기서 $p_{model}(h)$ 는 latent variables의 prior distribution 이다. 이 "prior" 는 위 discriminant model에서의 prior 와는 다른데, 그 prior는 우리가 트레이닝 데이터를 보기 전부터 이미 갖고 있는 모델의 파라메터에 대한 믿음 (belief) 을 나타낸다면 generative model에서의 prior 는 모델의 데이터 $x$ 에 대한 믿음 (belief) 을 나타낸다.
> 잘 이해는 안 감. 아무튼 generative model의 prior가 더욱 본질에 가까울 것 같음.

...

### 14.2.2 Denoising Autoencoders
Cost function (loss function) 에 sparsity penalty $\Omega$ 를 추가하지 말고, reconstruction error 를 다르게 계산하는 방법도 있다. DAE (Denoising AutoEncoder) 는 $L(x,g(f(\tilde{x})))$

...



**TODO:**

* 5.5. Bias and Variance
* 5.6. MLE
* 5.7. Bayesian Statistics