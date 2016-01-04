---
layout: post
title: "Machine learning models: Generative vs Discriminant, Parametric vs Nonparametric."
---

## Generative model vs Discriminant model
Generative model은 joint probability $P(x,y)$ 를 학습하고, bayes rule을 이용해서 $P(y|x)$ 를 추론한다. 반면 discriminant model은  conditional probability $P(y|x)$ 를 직접 학습한다. Generative model 은 $P(x,y)$ 를 학습하므로, 학습한 확률에 기반하여 (x, y) pair (data point) 를 샘플링 (생성) 할 수 있다. 

## Parametric model vs Non-parametric model
Parametric model과 non-parametric model은 좀 더 어려운 개념이다. parametric model은 데이터의 parameter의 수를 정한 후 학습한다. 대부분의 확률분포는 (우리가 배우는 확률분포는) parametric 이다. 모집단으로부터 표본을 관찰하고, 관찰 결과로부터 모집단의 parameter (모수) 를 추정한다. 반면 non-parametric model는 predict 를 위해 학습한 parameter 뿐만이 아니라 관찰한 데이터들을 사용할 수 있다. 대표적인 non-parametric model 로는 LDA (latent dirichlet analysis) 나 kNN 이 있다. non-parametric model 은 파라메터의 수가 무한한 infinitely parametric model 이라고 볼 수도 있다. 이러한 특징 때문에 non-parametric model 이 parametric model 에 비해 더욱 자유롭고, 복잡한 표현이 가능하다.



## References
* [stackoverflow; Difference between Generative, Discriminating and Parametric, Nonparametric Algorithm/Model](http://stackoverflow.com/questions/23821521/difference-between-generative-discriminating-and-parametric-nonparametric-algo)
* [stackoverflow; What is the difference between a Generative and Discriminative Algorithm?](http://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm)
* [Q; What is the Difference between a Nonparametric vs Parametric Method (Wei Ping) ](http://introml.blogspot.kr/2011/09/q-what-is-difference-between.html)
* [stackexchange; Parametric vs. Nonparametric](http://stats.stackexchange.com/questions/50141/parametric-vs-nonparametric)
* [What is the difference between the parametric model and the non-parametric model?](https://www.quora.com/What-is-the-difference-between-the-parametric-model-and-the-non-parametric-model)
