---
layout: post
title: "Machine Learning and Its Applications to Biology"
tags: ['논문']
date: 2015-03-20 00:10:00
---
# Machine Learning and Its Applications to Biology

2007년, Adi L. Tarca, Vincent J. Carey, Xue-wen Chen, Roberto Romero, Sorin Draghici.

## Introduction

머신러닝은 크게 두가지 측면을 갖고 있다.

  1. 컴퓨터 프로그래밍을 통해 classification/prediction task가 수행된다.
  2. 적은 human input만으로, classifier를 생성한다.

머신러닝과 biology는 밀접한 관계에 있어 왔고, 대표적으로 ANN(artificial neural network)이 생물학적 접근에서 나왔다.

이 튜토리얼은 크게 4개로 나눌 수 있다:

  1. 정의와 수학적 전제
  2. Supervised learning
  3. Unsupervised learning
  4. Methods and examples

## Main Concepts and Definitions

생물학적 컨텍스트에서 object-to-class mapping 예제를 살펴보자. 조직 유전자의 프로파일로 질병 그룹을 매핑할 수 있고, 단백질 시퀀스로 두번째 구조를 매핑할 수 있다.

**Example 1)**  
tissue gene expression profile to disease group: 조직 유전자의 프로파일로 질병 그룹을 예측  
**Feature:** 조직 샘플에서 개별 유전자의 expression level(발현 수준)을 측정.

**Example 2)**  
protein sequences to their secondary structures: 단백질 시퀀스로 단백질의 두번째 구조를 예측  
**Feature:** 단백질 시퀀스에서 해당 포지션에 amino acid symbol의 존재 유무

단백질 구조 분류와 같은 몇몇 어플리케이션에서는 몇몇 샘플만이 라벨링(class label이 지정되어 있다) 되어있다. 이런 케이스를 semi-supervised 라 한다. semi-supervised technique을 적용함으로써, 클래스 라벨이 존재하는 샘플들만 가지고 모델링을 했을 때보다 더 좋은 결과를 얻을 수 있다.

## Supervised Learning

### Error estimation

**resubstitution**: classifier를 모델링하기 위해 사용한 데이터를 또 사용해서 테스트하여 error rate를 측정하는 것을 resubstitution이라 한다. 당연히, 오버피팅 등 문제가 많다.  
**hold-out precedure**: 데이터를 트레이닝 셋과 테스트 셋 두가지로 나누는 방법.

그러나 biological data에서는, 일반적으로 데이터가 부족하기 때문에 이렇게 사치스럽게 데이터를 사용할 수 없다. 이에 사용하는 것이 **cross-validation method**다.

  * **leave-one-out(LOO) cross-validation method**: 샘플을 n개로 나눠서 하나를 test set으로 쓰고 나머지 n-1개를 training set으로 쓴다. 이 작업을 n개의 셋에 대해서 n번 반복.
  * **N-fold cross-validation method**: n개의 샘플 중 m개를 test set으로 사용하는 방법 (N=n/m)

이러한 크로스 밸리데이션 메소드는 약간의 bias를 가져오지만, 높은 variance를 보여준다.

### Type of classifiers

일반적인 클래지피케이션 어프로치에서는, 어트리뷰트들이 컨티뉴어스 밸류여야 하고, 각 클래스 c가 다변수 정규분포(multivariate normal distribution)을 따른다고 가정한다.  
$x \sim N(m_c, \Sigma_c),~m_c:mean,~\Sigma_c:~covariance~matrix$

이러한 multivariate-normal pdf(probability density function)를 사용하고 알 수 없는 모평균과 공분산 매트릭스를 샘플(표본)로부터 구한 표본평균과 표본 공분산으로 채움으로써, 각 클래스에 대한 판별함수(discriminant function)를 계산할 수 있다.  
$g_c(x)=-(x-m_c)\hat\Sigma_c^{-1}(x-m_c)^T-log(|\hat\Sigma_c|)$

이 판별함수는 pdf $p(\mathbf x|y=c)$와 같은 경향성을 보인다. 판별함수의 값이 커지면 pdf값도 커진다. 클래스에 대한 판별함수의 값은 class mean과 covariance matrix에 따라 달라지며, 새로운 오브젝트 z는 이 판별함수의 값이 가장 큰 클래스에 할당된다. 이러한 접근은 nonlinear(quadratic) class boundaries를 형성하며, 따라서 이를 quadratic discriminant rule 또는 Gaussian classifier라 한다.

이 quadratic classifier 대신에, 모든 클래스의 covariance matrix가 같다고 가정한다면, 전체에 대해서 하나의 pooled covariance matrix를 사용할 수 있다. 이러한 방법은 클래스당 샘플의 수가 적을 때 특히 유용하다 - 소수의 샘플로 covariance matrix를 추정하면 정확도가 많이 떨어지므로, 차라리 모든 샘플을 사용해서 하나의 covariance matrix를 만들어 공통으로 사용한다. 이렇게 만들어진 classifier는 클래스 바운더리로 hyperplane을 사용하므로, normal-based linear discriminant라 한다.

그런데 이보다 더 샘플이 적은 케이스가 있다. 피처(어트리뷰트)의 수와 샘플의 수가 비슷한 경우, normal-based linear discriminant로도 부족하다. 이런 케이스의 경우, covariance matrix에서 off-diagonal(diagonal을 제외한 나머지)을 모두 0으로 만들어버림으로써 피처간 covariation을 무시한다. 이러한 방법을 diagonal linear discriminant라고 하며, 다양한 microarray analysis 들에 대해서 더 좋은 결과를 내기 위해 사용된다.

최근에 발표된 regularized linear discriminant analysis는 피처의 수가 샘플의 수를 초과할 때 유용하다.

#### k-Nearst neighbor classifier

k-NN은 distance-based method이고, 바이오 데이터에 대해서는 다른 방법을 사용하는 듯 하다. 본문에서는 자세히 소개하지 않는다. Distance measures in DNA microarray data analysis, 2005년, Gentleman 외. pp. 189-208 참조.

new object **z**와 가까운 오브젝트 k개를 골랐을 때, 클래스 c에 속하는 오브젝트의 개수를 $n_c$ 라 하자. 이 때 k-NN의 discriminant function은 $g_c(x)=n_c$ 가 된다.

### Decision tree

피처를 따로따로 고려하기 때문에 본질적으로 suboptimal이지만, 결과가 이해하기 쉽다는 장점이 있다.

### Neural network

가장 널리 사용되는 뉴럴 네트워크 아키텍처는 fully connected, three-layered 스트럭쳐다. data mining concepts and techniques 책에서는 hidden layer가 하나고 output layer가 하나면 2-layer NN이라고 했었는데 여기서는 이렇게 히든레이어가 하나 있는 구조를 three-layered NN이라고 한다.

뉴럴 네트워크의 각 노드는, 엣지를 통해 들어오는 각 값들을 [logstic sigmoid](http://en.wikipedia.org/wiki/Sigmoid_function)1 를 통해 노멀라이징한다.

$$\sigma(z)={1 \over 1+\exp(z)}$$

트레이닝 프로세스가 에러를 최소화할 뿐만 아니라 네트워크의 weights 또한 최소화하려 한다면 위 식은 수정될 수 있다. _weights regularization_ 는, input의 작은 변화가 output에 큰 영향을 끼치는 것을 방지하여 모델의 generalization을 강화한다. 이 방법은 클래스간의 경계(바운더리)가 sharp하지 않다는 것을 가정한다.

### Support vector machine

모든 classification 알고리즘들이 test set을 통해 error rate를 계산하고 모델을 개선함으로써 generalization을 한다. 반면에 SVM은 decision boundary를 maximal marginal hyperplane으로 잡기 때문에 근본적으로 general한 알고리즘이다. 

책에서 본 내용과는 조금 다르게, SVM도 에러를 허용한다. 

$$\min_w ({1 \over 2}||w||^2+C\sum_{i=1}^{N_T}\xi_i)$$

xi($\xi$)가 에러를 의미하며 C는 유저가 지정하는 error control constant다. 즉, Least-square의 relax/exact form같이, relax SVM이라고 할 수 있을 것이며 C=0일때 exact가 된다.

이 optimization problem은 [quadratic programming problem](http://en.wikipedia.org/wiki/Quadratic_programming)을 적용해서 [dual problem](http://en.wikipedia.org/wiki/Duality_%28optimization%29)으로 변환시킬 수 있다. 

$$f(x)=sign(wx^T+b)=sign(\sum_i \alpha_iy_i(x_ix^T)+b)$$

이 식에서 $\alpha_i$는 support vector를 의미하고, dual problem을 풀어서 구할 수 있다.

linear inseperable, 즉 주어진 데이터에 맞는 hyperplane을 구할 수 없는 경우 데이터를 더 높은 차원으로 매핑한다. 이를 위해 사용하는 게 kernel function이다. 이러한 커널 펑션에는 또다른 파라메터가 들어갈 수 있고(radial basis function, 즉 RBF 같은 경우) 이는 모델을 만드는 과정에서 학습된다. 논문에서는, nonlinear SVM의 경우 C도 학습된다고 되어 있다.

## Dimensionality reduction

디멘션 리덕션은 퍼포먼스와 정확도 모두에 영향을 끼치지만, 일반적으로 퍼포먼스 쪽이 강조된다(고 생각한다). 그러나 논문의 문맥에서는, 바이오 데이터에서는 데이터의 디멘션 p가 샘플의 수 n보다 훨씬 클 수 있고, 이 경우에 모델의 정확도를 신뢰할 수 없게 된다는 측면에서 소개하고 있다. 즉, 상대적으로 퍼포먼스 보다 정확도를 개선하기 위한 용도로 사용된다. 

크게 두 가지의 줄기가 있는데,  
(1) 기존의 피처들을 통합해서 새 피처를 만든다: PCA(principal component analysis) 등  
(2) 기존의 피처들의 서브셋을 뽑는다: feature selection 등

p &gt;&gt; n 일 때 발생하는 가장 큰 문제는 overfitting이다. 일반적으로 p가 크면 모델의 파라메터의 수도 증가한다. 따라서 모델이 복잡해지고 오버피팅하게 될 확률이 높아진다. 특히 n이 작으면 더더욱. 

디멘션 리덕션이 해답이 될 수 있지만 동시에, 주의해서 사용해야 한다. 피처들은 서로 연관되어서 어떠한 정보를 제공하는데, 간단한 디멘션 리덕션 알고리즘은 피처를 독립적으로 고려하고 잘못된 리덕션이 발생할 수 있다.

통계적 패턴인식 분야에서는 feature selection을 두 가지 방법으로 접근한다:  
(1) filter method  
t-test와 같은 피처간 관련성에 대한 통계적 테스트는 피처를 필터링하기 위해 사용된다. 예를 들어, 암 예측을 위해서는 피처간 연관성을 고려하여 marker gene(표지 유전자) 가 사용된다(이게 끝?).

(2) wrapper method  
필터 메소드는 간단하고 빠르지만 피처를 독립적으로 고려하기 때문에 문제가 있다. 래퍼 메소드는 classifier의 결과의 정확도를 사용해서 피처를 각각 독립적으로 그리고 여러개를 한꺼번에 평가한다. 예를 들어, k-NN 메소드의 정확도를 사용해서 genetic algorithm을 가이드한다.  
래퍼 메소드는 컴퓨테이션이 많이 필요하다는 단점이 있지만, 가장 적합한 피처들을 찾아내고 피처간 시너지 이펙트를 찾아낸다.

## Unsupervised Learning / Cluster Analysis

클러스터링은 데이터를 탐색하는데 유용한 테크닉이다. 특히 microarray gene expression과 같은 고차원의 데이터일 경우에는 더욱 그렇다. 

### Overview of clustering algorithm

예를 들어, 마이크로어레이 실험에서 오브젝트는 다른 클러스터로 분류되는 조직 샘플일 수 있다. 즉, 마이크로어레이 실험을 하기 전에 클러스터링 테스트를 통해 조직 샘플들이 같은 클러스터인지 다른 클러스터인지 확인함으로써 샘플을 탐색해 볼 수 있다.

대표적인 클러스터링 알고리즘으로 k-means, hierarchical, PAM(partitioning around medoids)가 있다. PAM은 k-means와 달리 각 클러스터의 센터(mean)를 클러스터의 오브젝트로 잡는다. 이로 인해 outlier에 robust한 특성을 갖는다.

클러스터링 결과의 평가는 silhouetee(실루엣) method를 사용한다.  
$a(i)$ = the average dissimilarity of $i$ with all other data within the same cluster  
$b(i)$ = the lowest average dissimilarity of $i$ to any other cluster, of which i is not a member

즉 a(i)는 지금 할당된 클러스터가 얼마나 적절한지를 알아보는 메저고, b(i)는 다른 클러스터에 할당한다면 어떨까를 알아보는 메저다. 이 둘을 구해서 비교한다:

$$s(i)=\{\{b(i)-a(i)} \over {\max\\{a(i),b(i)\\}}}$$

s(i)의 값이 1에 가까울수록 b(i) &gt;&gt; a(i)라는 의미고, 이는 즉 잘 클러스터링 되었다는 것을 뜻한다. -1에 가까울수록 다른 클러스터에 할당하는 것이 맞다는 의미다. 0이 나온다면 이는 이 오브젝트가 두 클러스터의 경계에 있다는 것을 의미한다.

바이오 어플리케이션에서는, 피처와 샘플 둘다 클러스터링한다. 예를 들면, gene expression data(유전자 발현 데이터)는 조직 샘플과 다른 유전자들 둘다와 클러스터링 될 수 있다. k-means와 hierarchical methods 둘 다 사용할 수 있지만 local expression pattern을 발견하지 못할 수 있다 - 예를 들어 특정 컨디션에서는 유전자가 공동 발현되고, 다른 컨디션에서는 독립적으로 발현되는 경우. 이 문제를 해결하기 위해 사용하는 것이 biclustering method이다. 이 메소드는 유전자와 실험 컨디션을 동시에 클러스터링하여 local pattern을 찾아낸다. <http://www.cs.princeton.edu/courses/archive/spr05/cos598E/Biclustering.pdf> 참고.

* * *

  1. 로지스틱 펑션의 일종으로 S자 커브를 그리는 함수↩


[Tistory 원문보기](http://khanrc.tistory.com/88)
