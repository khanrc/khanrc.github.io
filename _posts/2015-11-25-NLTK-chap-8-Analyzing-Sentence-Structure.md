---
layout: post
title: "NLTK chap 8. Analyzing Sentence Structure"
tags: ['DataScience/Text Mining']
date: 2015-11-25 21:30:00
---
# [Analyzing Sentence Structure](http://www.nltk.org/book/ch08.html)

### 1\. Some Grammatical Dilemmas

문장은 문맥에 따라 얼마든지 다르게 해석될 수 있다. 대표적인 예가 "time flies like an arrow" 이다. 사람은 보통 이 문장을 시간은 화살처럼 흐른다 라고 해석하겠지만, 문법적으로는 시계파리는 화살과 같다 라는 해석 또한 가능하다.

![ambiguity1](http://www.nltk.org/book/tree_images/ch08-tree-1.png)   
![ambiguity2](http://www.nltk.org/book/tree_images/ch08-tree-2.png)

### 2\. What's the Use of Syntax?

![constituent](http://www.nltk.org/book/tree_images/ch08-tree-3.png)

위 그림에서 각 노드를 constituent (구성요소) 라 한다. 한 문장은 여러개의 constituent로 나뉘어지고, 그 constituent는 또 다른 constituent들로 나뉘어진다. 

### 3\. Context-Free Grammar

CFG (Context-Free Grammar) 란, 문맥에 상관없이 적용가능한 문법을 말한다. 다음과 같은 형태를 띈다:
    
    
      S -> NP VP
      VP -> V NP | V NP PP
      PP -> P NP
      V -> "saw" | "ate" | "walked"
      NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
      Det -> "a" | "an" | "the" | "my"
      N -> "man" | "dog" | "cat" | "telescope" | "park"
      P -> "in" | "on" | "by" | "with"

실상 자연어에는 적용하기 어려우며, 프로그래밍 언어는 CFG 에 기반하여 표현할 수 있다.

### 4\. Parsing with Context-Free Grammar

CFG는 프로그래밍 언어에 주로 적용되는 개념이므로, 컴파일러에서 이러한 개념을 많이 사용한다. 따라서 CFG를 파싱하는 기법 또한 컴파일러에서 사용된다. CFG의 파싱은 크게 Recursive descent parsing으로 대표되는 Top-down method와 Shift-reduce parsing으로 대표되는 Bottom-up method가 있다. 

둘 다 Brute-Force 라고 할 수 있는 backtracking 방법론으로, Top-down method는 전체 문장으로부터 시작하여 점점 쪼개어 나가는 방법이고 Bottom-up method는 단어단어의 품사를 결정하고 하나씩 합쳐서 전체 문장을 만드는 방식이다.

이 두가지 외에 유명한 파서로 두 가지 파서가 소개되는데, 1) Left-Corner Parser는 Top-down parser인데 bottom-up filtering을 수행하는 파서이고 2) Chart Parser는 DP (dynamic programming) 를 이용하여 수행속도를 빠르게 한 파서다.

### 5\. Dependencies and Dependency Grammar

문장에는 의존성 (dependency) 이 있다.

![dependency](http://www.nltk.org/images/depgraph0.png)

SBJ는 subject, OBJ는 object, NMOD는 noun modifier 이다.

### 6\. Grammar Development

#### Pernicious Ambiguity (치명적인 모호성)

자연어를 CFG로 표현하기 위해서는 굉장히 복잡한 CFG가 필요하다. 그걸 어떤 코퍼스에 대해 만들었다고 하더라도, 문장의 길이에 따라 그 CFG로 표현할 수 있는 parse tree의 수는 천문학적으로 많아진다. 즉, 복잡한 자연어를 표현할 수 있는 CFG라면, 어떤 문장에 대해 그 문장을 해석할 수 있는 parse tree를 수많은 방법으로 구성할 수 있다는 것이다. 즉, 현실적으로 자연어를 CFG로 표현하는 것은 불가능하다.

#### Weighted Grammar

이러한 문제를 해결하기 위해, PCFG (probabilistic context-free grammar) 가 등장했다. 이 방법은 각 constituent의 분화에 확률을 부여함으로써 확률적 추정이 가능하게 한다.
    
    
        S    -> NP VP              [1.0]
        VP   -> TV NP              [0.4]
        VP   -> IV                 [0.3]
        VP   -> DatV NP NP         [0.3]
        TV   -> 'saw'              [1.0]
        IV   -> 'ate'              [1.0]
        DatV -> 'gave'             [1.0]
        NP   -> 'telescopes'       [0.8]
        NP   -> 'Jack'             [0.2]

위와 같이 확률을 배정할 수 있으며 이는 NLTK에서는 이를 `nltk.ViterbiParser(grammar)` 로 파싱할 수 있다.


[Tistory 원문보기](http://khanrc.tistory.com/130)
