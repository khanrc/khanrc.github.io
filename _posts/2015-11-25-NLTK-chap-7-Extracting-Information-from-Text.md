---
layout: post
title: "NLTK chap 7. Extracting Information from Text"
tags: ['DataScience/Text Mining']
date: 2015-11-25 21:28:00
---
# [Extracting Information from Text](http://www.nltk.org/book/ch07.html)

요즘 자연어 처리 특론 수업을 듣는데, [NLTK 책](http://www.nltk.org/book/)으로 수업을 한다. 지금까지는 딱히 정리할 만한 내용이라고 못 느껴서 정리를 안 했는데 이번 챕터는 살짝 정리해 본다. NLTK 책에 더욱 자세하게 코드와 함께 설명되어 있다. 여기서는 개념만 간단하게 짚는다.

### 1\. Information Extraction

말 그대로 Text로부터 정보를 추출하는 것을 의미한다.

![overview](http://www.nltk.org/images/ie-architecture.png)

위와 같은 과정으로 수행된다. 1) 텍스트를 문장으로 조각내고 2) 단어 단위로 토크나이즈 하고 3) POS tagging을 통해 품사를 구분하고 4) Named entity를 찾아내고 5) 관계를 분석한다. 여기서 1, 2, 3단계는 지금까지 책에서 설명했고, 여기서는 그 이후를 다룬다. NER (name entity recognition) 을 위한 첫 스텝은 chunking이다.

### 2\. Chunking

![chunking](http://www.nltk.org/images/chunk-segmentation.png)

청킹이란, 위 그림과 같이 POS tagging 이후 단어들을 구 (pharse) 형태, 즉 청크로 묶는 것이다. 

#### NP-Chunking

Noun Phrase Chunking.   
![np-chunking](http://www.nltk.org/book/tree_images/ch07-tree-1.png)

WSJ (Wall Street Journal) Corpus에 있는 NP-Chunk 는 이렇게 생겼다:
    
    
    [ The/DT market/NN ] for/IN [ system-management/NN software/NN ] for/IN [ Digital/NNP ] [ 's/POS hardware/NN ] is/VBZ fragmented/JJ enough/RB that/IN [ a/DT giant/NN ] such/JJ as/IN [ Computer/NNP Associates/NNPS ] should/MD do/VB well/RB there/RB ./.

### Chinking

칭크 (Chink) 는 문장에서 청크를 제외한 나머지 부분을 의미한다. 칭킹이란, 문장에서 청크를 찾는 청킹과는 반대로 문장에서 칭킹을 찾아 제외함으로써 최종적으로 청크를 찾는 방법이다.

### Representing Chunks: Tags Versus Trees

이러한 청크들을 나타내는 방법으로 크게 두가지가 있는데, 첫번째는 Inside, Outside, Begin 으로 청크를 구분하는 IOB tags 방법이다.

![IOB tags](http://www.nltk.org/images/chunk-tagrep.png)

위 그림과 같이 청크의 시작 단어는 B-NP, 청크의 시작 단어가 아닌 내부 단어는 I-NP, 청크가 아닌 칭크는 O 로 태깅하는 방법이다.

NLTK에서는 이러한 방법을 사용하지 않고, 이렇게 태깅된 코퍼스가 있으면 트리 구조로 변환한다. 트리 구조가 더욱 효율적으로 사용할 수 있다.

![tree](http://www.nltk.org/images/chunk-treerep.png)

### 3\. Developing and Evaluating Chunkers

#### Present Chunckers

  * The Regular Expression-Based Chunkers   

    * 말 그대로 Regex 기반으로 Chunk를 구분함. Rule-based system.
  * The N-gram Chunkers   

    * N-gram base 로 학습. 아마 그냥 통계적으로 제일 많이 나온 tag-chunk 조합을 할당하는 듯
  * Classifier-Based Chunkers   

    * train_set으로부터 피처를 뽑고 머신러닝을 통해 classifier를 학습. Machine-learning system.

#### CoNLL-2000 Chunking Corpus

WSJ 텍스트의 27만개 단어로 구성된 청킹 코퍼스. IOB 포맷으로 청킹되어 있고 POS tagging이 되어있으며 train/test set으로 나뉘어져 있다. 이 코퍼스는 NP 청크, VP 청크, PP 청크 총 3가지 청크를 가진다.

### 4\. Recursion in Linguistic Structure

위의 청커들은 전부 1단계에서만 수행한다. 즉, 청크 안에 청크가 있는 형태를 분석할 수 없다. 이러한 Nested structure를 해석하기 위해 Cascaded Chunker는 구조를 재귀적으로 분석한다. 

![nested structure](http://www.nltk.org/book/tree_images/ch07-tree-3.png)

### 5\. Named Entity Recognition

  


![](http://cfile2.uf.tistory.com/image/2415184F5655A9A401F5BC)

  


NER의 최종 목표는 모든 네임드 엔티티들을 식별하는 것이다. 이 작업은 두가지 서브태스크로 나눌 수 있는데 1) NE들을 찾아내고 2) NE의 타입을 식별한다.

NER이 쉬워 보일 수 있는데, 생각처럼 쉽지 않다. 다음은 location detection을 수행한 것이다:

![NER difficulties](http://www.nltk.org/images/locations.png)

즉 on이나 books같은 일반적인 단어들이 지명 (location name) 에 포함된다는 문제가 있는 것이다. nltk에서는 학습된 NE chunker `nltk.ne_chunk()`를 제공한다. 

태깅, 청킹, NER 모두 스탠다드 툴이 제공되지만, 바이오 텍스트 마이닝 등 특정 분야에 대해 텍스트 마이닝을 수행할 경우 그에 알맞는 코퍼스를 사용하여 태거, 청커, Recognizer를 학습하여 쓰는 것이 더 좋은 결과를 낼 것이다.

### 6\. Relation Extraction

관계 추출은 특정한 방법론이 존재하지 않고 코퍼스에 알맞는 방법론을 그때그때 사용하는 듯 하다. 책에 별다른 설명이 없음. 자주 쓰이는 방법론 중 하나는, 모든 (X, a, Y) 구조를 찾는다. 여기서 X, Y는 네임드 엔티티이고, a는 두 엔티티간의 관계를 나타낸다. 그 이후 관계를 나타내는 a들을 분석하여 관계를 추출할 수 있다.

  


![](http://cfile3.uf.tistory.com/image/242F6A4D5655A9BB142BE3)

  



[Tistory 원문보기](http://khanrc.tistory.com/129)
