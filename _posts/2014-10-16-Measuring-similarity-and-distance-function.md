---
layout: post
title: "Measuring similarity and distance function"
tags: ['DataScience']
date: 2014-10-16 19:08:00
---
# [Measuring similarity and distance function](http://horicky.blogspot.kr/2012/08/measuring-similarity-and-distance.html)

이 글은 위 링크의 번역+@이다.

두 데이터 포인트간의 similarity나 distance를 측정하는 것은 `kNN`이나 `Clustering` 등 많은 머신러닝 알고리즘의 기본이다. 데이터 포인트의 특성에 따라, 다양한 측정법을 사용할 수 있다.

## Distance between numeric data points

### Minkowski distance

$$\sqrt[p]{(x_1 - x_2)^p + (y_1 - y_2)^p}$$

when p=1: `Manhattan distance`  
when p=2: `Euclidean distance`

이 측정법은 데이터 분포에 독립적이다. 하지만 x축의 값들이 y축보다 훨씬 크다면 어떨까? 이런 문제들 때문에 우리는 먼저 각각의 컬럼들을 같은 스케일로 맞출 필요가 있다. 일반적으로는 `z-transform`을 사용한다: 각각의 데이터 포인트1를 평균으로 빼고 표준편차로 나눈다.

$$z-transform:~(x_1, y_1)~becomes~(\dfrac{x_1-μ_x}{σ_x}, \dfrac{y_1-μ_y}{σ_y})$$

### Mahalanobis distance

이 측정법은, 비록 각 디멘션의 분포를 고려했다 하더라도, 각 디멘션들이 서로 독립이라고 가정한다. 하지만 x-디멘션과 y-디멘션이 서로 `correlation`이 있다면 어떨까? 이렇게 각 디멘션간의 `correlation`을 고려하기 위해, 우리는 이 디스턴스 measure를 사용할 수 있다:

$$\sqrt{(\vec v_1 - \vec v_2)^\top \Omega (\vec v_1 - \vec v_2)}$$

### Consine distance

![Cosine distance](http://1.bp.blogspot.com/-KbVxVkhYDUw/UB8i5eD0bOI/AAAAAAAAAyU/gZgdACaaauM/s320/p1.png)  
만약 우리가 데이터의 크기보다 방향을 더 고려한다면, `cosine distance`가 일반적인 접근이다. 코사인 디스턴스는 두 데이터 포인트의 dot product를 크기의 곱으로 나눈다.

$$cosine~similarity = \dfrac{AB}{|A||B|}$$

[term/document matrix](http://en.wikipedia.org/wiki/Document-term_matrix)와 함께, `cosine distance`는 도큐먼트간의 유사도(=distance)를 측정하기 위해 일반적으로 사용된다.

> distance와 similarity를 헷갈리지 말 것. cosine distance = 1 - cosine similarity.

## Distance between categorical data points

### Simple matching coefficient

categorical value는 순서가 없기 때문에, 우리는 오직 categorical value가 같은지 아닌지만 측정할 수 있다. 기본적으로 attribute value가 겹치는 정도를 측정한다. `Hamming distance`는 두 데이터 포인트가 서로 매치되려면 얼마나 많은 attribute가 바뀌어야 하는지를 측정한다. 두 데이터포인트의 유사도를 결정하기 위해 `simple matching coefficient`를 사용할 수 있다.

$$simple~matching~coefficient = \dfrac{\\#(Match Attributes)}{\\#(Attributes)}$$

(# = number of)

### Jaccard similarity coefficient

그러나, 데이터 포인트가 비대칭 바이너리 데이터 attribute를 가지고 있다면, 어떤 value의 equality는 더이상 의미가 없다. 예를 들어, 데이터 포인트가 유저를 나타내고 attribute가 각 영화를 나태낸다고 해보자(즉 각 row가 유저이고 column이 영화가 된다). 그리고 1이 yes, 0가 no라고 하자. 대부분의 유저들이 전체 영화의 양에 비하면 굉장히 적은 양의 영화를 보았다는 점을 고려하면, 두 유저가 둘다 어떤 영화를 보지 않았다는 것이(즉 value가 둘다 0이라는 것이) 두 유저가 유사하다는 것을 의미하지는 않는다. 반면에 두 유저가 같은 영화를 보았다면(즉 value가 둘다 1이라면), 이는 두 유저가 매우 유사하다는 것을 의미한다. 이러한 경우에, 둘 다 1이라는 것은 둘 다 0인 경우에 비해 훨씬 중요하다. 이는 `Jaccard similarity`로 연결된다:

$$jaccard~similarity = \dfrac{|A \cap B|}{|A \cup B|} = \dfrac{\\#(1~In~A~and~B)}{\\#(1~In~A) + \\#(1~In~B) - \\#(1~In~A~and~B)}$$

### etc

이러한 matching or not 이외에, 만약 카테고리가 트리구조로 계층화되어있다면, 두 카테고리간의 거리를 그들의 공통 조상까지의 path length로 나타낼 수 있다. 예를 들어, "/product/spot/ballgame/basketball"은 "/product/luxury/handbags"보다 "/product/spot/ballgame/soccer/shoes"에 가깝다.

## Similarity between instances containing mixed types of attributes

데이터 포인트가 mixed of attributes를 가질 때, 즉 categorical attribute와 numerical attribute가 둘다 존재할 때를 말한다. 이 때 각 어트리뷰트의 유사도(또는 같은 타입의 어트리뷰트 그룹의 유사도)를 측정할 수 있고, 이 결과를 `weighted average`2를 이용하여 통합할 수 있다. 

하지만 의미없는 비대칭 attribute를 다룰 때는 조심해야 한다. 

\\(combined\\_similarity(x,y) = \dfrac{\sum over\\_k[w_k * δ_k * similarity(x_k, y_k)]}{\sum over\\_k(δ_k)}\\\ where ~\sum over\\_k(w_k) = 1\\)

> 이 식이 뭔지 잘 모르겠다. 아무튼 `weighted average`로 통합할 수 있다.

## Distance between sequence (String, TimeSeries)

### String - [Edit distance](http://en.wikipedia.org/wiki/Levenshtein_distance)

각 attribute가 sequence의 element를 나타내는 경우, 거리를 다른 방식으로 측정할 필요가 있다. 예를 들어, 각 데이터 포인트가 스트링(sequence of characters)이라고 해 보자. 이 경우에는 `edit distance`가 일반적인 측정 방법이다. 기본적으로, `edit distance`는 stringA를 stringB로 변환하려면 얼마나 많은 "modifications" (insert, modify, delete) 이 필요한지를 의미한다. 이는 `dynamic programming`으로 구할 수 있다.

### TimeSeries - [Dynamic Time Warp](http://en.wikipedia.org/wiki/Dynamic_time_warping)

Time Series는 시퀀스 데이터의 또다른 예다. 에딧 디스턴스의 컨셉과 비슷하게, `Dynamic Time Wrap`은 두 타임 시리즈에 데이터 포인트를 추가함으로써 타임 디멘션의 뒤틀림에 대해 스퀘어 에러를 최소화하는 방법이다. 이 때 데이터 포인트를 어디에 추가해야 하느냐 또한 유사한 `dynamic programming`으로 구한다. 여기 이에 대한 아주 좋은 논문이 있다: [Computing and Visualizing Dynamic Time Warping Alignments in R: The dtw Package](http://www.jstatsoft.org/v31/i07/)

> 무슨 소린지 이해하기 힘들다. 가능하면 본문을 읽어보고 위키와 논문을 참고하도록 하자.

## Distance between nodes in a network

`homogeneous undirected graph`, 즉 엣지에 방향이 없고 노드가 전부 같은 타입의 그래프에서 노드간 거리는 `shortest path`로 측정될 수 있다.

[`bi-partite graph`](http://en.wikipedia.org/wiki/Bipartite_graph)에서는 두 타입의 노드가 있고 엣지는 서로 다른 노드를 잇는다. 이 때 같은 타입의 노드간의 similarity는 그들이 연결된 다른 타입의 노드들이 얼마나 비슷한가를 분석함으로써 측정할 수 있다.

### SimRank

`SimRank`는 연결된 두 노드간의 similarity를 합하여 각 노드 타입의 유사도를 계산하는 반복적인 알고리즘이다.  
![SimRank](http://4.bp.blogspot.com/-xEeArTG2Bt4/UB9inuJqfWI/AAAAAAAAAy4/G3plxOjwUl4/s400/p2.png)

### RandomWalk

similarity를 계산하기 위해 `RandomWalk`와 같은 확률적 접근을 사용할 수도 있다. 각 people node는 사람의 이름이 적힌 token을 랜덤하게 선택한 community node로 보낸다. 각 community node는 마찬가지로 받은 토큰을 랜덤하게 선택한 people node로 보낸다. 이제 토큰을 되돌려받은 people node는 β의 확률로 토큰을 drop하거나, 또는 다시 랜덤하게 community node를 선택하여 되돌려보낸다. 모든 토큰이 drop되어 사라질 때까지 이 프로세스를 반복한다. 그리고 나면, trace matrix를 얻을 수 있고 각 노드가 받은 토큰들의 dot product로 similarity를 계산할 수 있다.

![RandomWalk](http://4.bp.blogspot.com/-7tw_AarlFsE/UB_xkudBSHI/AAAAAAAAAzU/hPWFSSiP6I4/s400/p3.png)

## Distance between population distribution

각 데이터 포인트간의 distance를 계산하는 거 외에도, 우리는 데이터 포인트들의 collection을 비교하고 distance를 측정할 수 있다. 대표적으로 population이 그러하다. 사실, 통계학의 중요한 파트다: 두 샘플 그룹간의 distance를 측정하고 그 두 그룹이 서로다른 population으로부터 나왔다고 결론내릴 수 있을 만큼 유의미한 차이를 갖는지 판단하는 것이.

population이 서로다른 카테고리에 속한 멤버들을 포함하고, 우리가 population A와 population B가 이 카테고리들에 걸쳐 같거나 다른 멤버 비율을 갖는다면, distance를 측정하기 위해 [`Chi-Squared test`](http://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test)이나 [`KL-Divergence`](http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)를 쓸 수 있다.

> [카이제곱 분포](http://ko.wikipedia.org/wiki/%EC%B9%B4%EC%9D%B4%EC%A0%9C%EA%B3%B1_%EB%B6%84%ED%8F%AC), [쿨백-라이블러 발산](http://ko.wikipedia.org/wiki/%EC%BF%A8%EB%B0%B1-%EB%9D%BC%EC%9D%B4%EB%B8%94%EB%9F%AC_%EB%B0%9C%EC%82%B0). 확률분포의 차이를 계산한다.

### [Pearson Correlation coefficient](http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient)

population의 모든 멤버가 두 서로다른 numeric attributes를 갖는다면 (예를 들어 weight, height 라던가), 그리고 그 두 attribute가 correlated되었는지 궁금하다면, `correlation coefficient`가 correlation 정도를 측정해 줄 것이다. 이 두 attribute가 같은 방향으로 같이 가든지(몸무게가 많이 나갈수록 키가 크다), 반대로 가든지(몸무게가 많이 나갈수록 키가 작다), 또는 독립적이든지. `correlation coefficient`는 -1(negatively correlated)에서 0(no correlation)을 거쳐 1(positively correlated)까지 있다.

### [Mutual information](http://en.wikipedia.org/wiki/Mutual_information)

만약 두 어트리뷰트가 categorical이라면, correlation을 측정하기 위해 `mutual information`이 주로 쓰인다.

### [Spearman's rank correlation coefficient](http://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)

어트리뷰트의 랭킹을 기반으로 correlation을 측정하고 싶다면, `Spearman's rank correlation coefficient` 를 사용하라. 스피어만 상관계수는 데이터가 서열척도인 경우, 즉 자료의 값 대신 순위를 이용하는 경우의 상관계수다. 

## Addition: algorithm

`SVM`, `Random Forest`, `Random Ferns` 등 전부 distance method가 아니라 machine learning algorithm이다. 지금까지 얘기한 distance method와는 다른 차원이란 소리.

### [Support Vector Machine](http://ko.wikipedia.org/wiki/%EC%84%9C%ED%8F%AC%ED%8A%B8_%EB%B2%A1%ED%84%B0_%EB%A8%B8%EC%8B%A0)

`SVM`은 데이터셋을 분리하는 `hyperplane`중에서 가장 거리가 먼(maximum-margin) `hyperplane`을 찾는다.

### [Random Forest](http://rstudio-pubs-static.s3.amazonaws.com/4239_fcb292ade17648b097a9806fbe026e74.html), [Random Ferns](http://darkpgmr.tistory.com/90)

두 데이터포인트나, population을 비교하는 지금까지의 distance or similarity 메소드와는 달리, `Random Forest`나 `Random Ferns`는 `Decision Tree`기반의 Machine Learning 알고리즘이다.

[Random Ferns 소개](http://randomferns.blogspot.kr/2014/03/random-ferns.html)

* * *

  1. data point란 데이터셋에서 하나의 데이터, 즉 하나의 row를 의미한다.↩

  2. 가중평균. 각 값에 weight를 부여하여 평균을 낸다↩

  



[Tistory 원문보기](http://khanrc.tistory.com/60)
