---
layout: post
title: "Bayesian probability: MLE, MAP"
tags: ['DataScience']
date: 2015-07-01 11:49:00
---
# Bayesian probability

likeihood, postierior, prior, MAP, MLE 등등 맨날 나오는 데 뭔지 잘 몰라서 공부한다.

모집단으로부터 표본을 관찰하는 것: 확률   
우도(likelihood)는 확률의 반대 개념. 관찰한 표본을 기반으로 이 표본이 추출될 가능성이 가장 높은(즉 우도가 가장 높은) 모집단을 추정하는 방법이 최대우도법(MLE, maximum likelihood estimation)이다.

![철수영희](http://cfile5.uf.tistory.com/image/2368DA4A52D8916E3B67E3)

커튼에 비친 사람의 형상을 관찰할 수 있다고 하자. 이 때 조건부 확률로 이 상황을 나타내는 방법은 두가지다. P(형상|철수), P(철수|형상). 실제로 우리가 알고싶은 것은 이 형상이 철수일 확률로, P(철수|형상)이다. 이를 posterior라 한다. 일반적으로 posterior는 알기 힘들고, P(형상|철수)를 아는 것이 더욱 쉽다. 이를 likelihood라 한다. 

이 두가지가 조금 헷갈릴 수 있는데, posterior P(철수|형상)은 이 형상이 철수일 확률로 형상이 있을 때 철수일 확률이다. likelihood P(형상|철수)는 철수가 이 형상으로 나타났을 확률로 철수가 가질 수 있는 다양한 형상들 중 눈앞의 커튼에 비친 형상일 확률이다. 

이 두가지 확률은 베이즈 정리로 묶일 수 있다.

<div>$P(철수|형상)=\dfrac{P(형상|철수)P(철수)}{P(형상)}$</div>

여기서 결국 P(형상)은 고정된 상수값이므로 

<div>$P(철수|형상) \propto P(형상|철수)P(철수)$</div>

이고, 이를 일반적으로 표현하면

<div>$P(posterior) \propto likelihood \times P(prior) $</div>

이다. P(철수), P(영희) 따위의 클래스 고유의 확률값을 prior 확률이라 한다.

## 통시적(diachronic) 해석

이 posterior, likelihood, prior는 Think bayes책의 1.5장 통시적 해석을 보면 좀 더 쉽게 이해할 수 있다. 베이즈 정리를 H와 D에 대해 쓰면

<div>$P(H|D)=\dfrac{P(H)P(D|H)}{P(D)}$</div>

이다. 짐작컨대, H는 Hypothesis이고 D는 Data인 듯 하다. 이 해석에서 각각의 기호의 의미는 다음과 같다:

  * P(H)는 데이터를 보기 전의 가설의 확률로, **사전 확률(prior)**이라 한다.
  * P(H|D)는 여기서 계산하고자 하는 데이터를 확인한 이후의 가설 확률로, **사후 확률(posterior)**이라 한다.
  * P(D|H)는 데이터가 가설에 포함될 확률로, **우도(가능도, likelihood)**라 한다.
  * P(D)는 어떤 가설에든 포함되는 데이터의 비율로, **한정 상수(normalizing constant)**라 한다.

### In machine learning context

머신러닝의 맥락에서, 일반적으로 이는 

<div>$P(C|D)=\dfrac{P(C)P(D|C)}{P(D)}$</div>

가 된다. posterior를 구하기 위해서 베이즈 정리를 사용한다. prior는 각 클래스의 등장 비율이다. likelihood는 일반적으로 가장 구하기 쉽다고 하는데, 각 클래스에서 해당 데이터의 등장 비율이다. normalizing constant P(D)는 구하기 까다로울 수 있다고 하는데, 클래스와 관계없이 전체 데이터셋에서 해당 데이터의 등장 비율이다. 책을 보면 자세히 나오지만, 이 한정 상수를 구하려면 임의의 데이터가 항상 어떤 클래스를 갖고(어느 클래스에도 안 속하는 경우가 없고), 여러 클래스를 동시에 가질 수 없다는 가정이 들어가는 것이 일반적이다.

## MLE, MAP

이 형상이 누구인지를 확률적으로 알아내는 대표적인 방법으로 MLE(maximum likelihood estimation)와 MAP(maximum a posterior)가 있다. MLE는 likelihood를 모두 구해 최대값을 선택하는 방법으로 P(형상|철수), P(형상|영희) … 등을 다 구해 가장 확률이 큰 클래스를 선택하면 된다. MAP는 여기에 prior를 더해 posterior를 구한 것으로, P(철수|형상), P(영희|형상) … 등을 구해 클래스를 선택한다.

MAP는 결국 MLE에 prior를 더하는 것이고, 반대로 MLE는 MAP에서 모든 P(prior)가 같다는 가정이 들어가는 것이다.

최종적으로, 식은 아래와 같다(c: 클래스, z: 관측한 샘플):

![MLE, MAP](http://cfile3.uf.tistory.com/image/246D0C4852D8CCE5121E08)

## pure, naive, semi-naive Bayesian

지금까지 P(D)라고 간단하게 이야기했지만, 정확히 말하면 보통 이 D는 여러가지 특성(feature)들로 구성되어 있다. 즉, 

<div>$D=(d_1, d_2, ... , d_n)$</div>

위에서 언급했듯이, P(D)를 구하려면 전체 데이터셋에서 D와 동일한 데이터의 등장 비율을 구하면 된다(물론 정확하게는 모집단에서 D가 등장할 확률을 구해야 한다). 이를 `pure bayes`이라 한다.

그런데, 현실적으로 `pure bayes`를 구하기는 어렵다. 각 데이터에 대해 분포를 다 계산하기가 어렵다는 것이 첫번째 이유이고, 둘째로 데이터의 특성 수 n이 증가할수록 동일한 데이터는 줄어들게 되는 문제가 있다. 예를 들어 사람에 대한 데이터셋이라면, 키, 몸무게, 신체치수, 나이, 머리 색, 눈 색, 눈코입의 크기 등 이러한 특성값들이 전부 동일한 데이터는 존재하지 않을 것이다. 

이러한 현실적인 문제를 극복하고자 등장한 것이 `naive bayes`이다. `naive bayes`는 각 특성들이 독립이라는 가정 하에 수행된다. 각 특성들이 독립이라면

<div>$P(D)=P(d_1)P(d_2)...P(d_n)$</div>

이 된다. 특성들이 전부 독립이라는 가정이 맞다면 이는 `pure bayes`와 동일한 결과를 보여준다.

마지막으로 이 두가지의 중간 단계에 있는 `semi-naive bayes`가 있다. `semi-naive bayes`는 관련이 있는 특성들을 모아 그룹핑을 하여 `pure bayes`로 계산하고, 이렇게 묶인 그룹끼리는 서로 독립으로 즉 `naive bayes`로 계산한다.

> `semi-naive bayes`의 계산방법은 틀렸을 수 있음. 정확한 방법이 필요하면 따로 찾아보도록 하자.

이 세가지 베이지안 확률 모델을 수식으로 정리하면 다음과 같다:   
![pure](http://cfile22.uf.tistory.com/image/2767804652D8C9F91842D9)   
![naive](http://cfile7.uf.tistory.com/image/2720944252D8C9A919FE26)   
![semi-naive](http://cfile29.uf.tistory.com/image/25217C4452D8C9B9239D60)

## Negative log-likelihood

classifier의 정확도를 판단할 때 negative log-likelihood를 사용하곤 한다.   
log를 사용하는 맥락은 크게 세가지인데,

  1. 매우 작은 likelihood에 따라 발생하는 underflow의 발생 가능성을 감소시킨다.
  2. 곱하기가 더하기로 바뀌는 addition trick이 적용된다.
  3. log 함수는 monotone transformation이다. (단조함수)

MLE에서,

<div>$\mathcal L(\theta|x_1,...,x_n)=f(x_1,x_2,...,x_n|\theta)=\prod_{i=1}^n f(x_i|\theta)$</div>

이다. 여기서 벡터 x는 sample이고, theta는 클래스다.

> $\mathcal L(A|B)=P(B|A)$는 B가 주어졌을 때 A의 likelihood이다.

<div>$\log{\mathcal L(\theta|x_1,...,x_n)}=\sum_{i=1}^n \log{f(x_i|\theta)}$</div>

이고 여기서 maximum likelihood estimtor는 아래와 같다:

<div>$\hat\theta_{MLE}=\mathop{\arg\,\max}\limits_\theta \sum_{i=1}^n \log f(x_i |\theta)$</div>

[arg max](http://egloos.zum.com/etstnt/v/5222752)란, argument of maximum의 약자로 max값을 구했을 때의 argument의 값을 의미한다. 위에서는 theta. 즉, 위 식을 풀어 설명하면 클래스를 바꿔가며 log likelihood를 구했을 때 그 최대값을 찾는 MLE이다.

> 참고로 이 MLE는 MAP로 바꿀 수도 있는데, uniform prior $g(\theta)$를 넣으면:
> 
> $\hat\theta_{MLE}=\mathop{\arg\,\max}\limits_\theta \sum_{i=1}^n \log f(x_i |\theta)=\mathop{\arg\,\max}\limits_\theta \log(f|\theta)=\mathop{\arg\,\max}\limits_\theta\log(f|\theta)g(\theta)=\hat\theta_{MAP}$
> 
> 가 된다.

그리고 위 식은 아래와 같이 바꿀 수 있다:

<div>$\mathop{\arg\,\max}\limits_\theta \sum_{i=1}^n \log f(x_i |\theta)=\mathop{\arg\,\min}\limits_\theta -\sum_{i=1}^n \log f(x_i |\theta)=\hat\theta_{MLE}$</div>

즉, negative log likelihood가 MLE를 나타낼 수 있다! [원문](https://quantivity.wordpress.com/2011/05/23/why-minimize-negative-log-likelihood/)을 보면, KL-divergence를 이용해서 무언가를 더 설명하는데 이해가 잘 안 감. 굳이 negative로 바꾸는 이유는, optimization problem이 전부 minimum을 구하기 때문에, 여기서도 negative로 바꿔줘야 arg max가 arg min으로 바뀐다.

## Reference

[다크 프로그래머: 베이지언 확률(Bayesian Probability)](http://darkpgmr.tistory.com/119)   
[Why Minimize Negative Log Likelihood?](https://quantivity.wordpress.com/2011/05/23/why-minimize-negative-log-likelihood/)   
Think Bayes: 파이썬을 활용한 베이지안 통계 (책)


[Tistory 원문보기](http://khanrc.tistory.com/97)
