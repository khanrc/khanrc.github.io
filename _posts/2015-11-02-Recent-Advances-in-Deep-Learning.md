---
layout: post
title: "Recent Advances in Deep Learning"
tags: ['Deep Learning']
date: 2015-11-02 22:42:00
---
**Recent Advances in Deep Learning and Introduction to VUNO**

SK Tech Planet 2015 발표 정리 (VUNO 에 대한 내용은 정리하지 않음)

동영상: <https://www.youtube.com/watch?v=-IeJiZw189c>

슬라이드: <http://readme.skplanet.com/wp-content/uploads/3-7-%EC%A0%95%EA%B7%9C%ED%99%98_VUNO_Recent-Advances-in-Deep-Learning-and-Introduction-to-VUNO.pdf>

슥슥 보면서 빠르게 정리했기에 빠진 내용도 있을 수 있고 설명이 그다지 자세하지 않아서 이해를 잘못한 부분도 있을 수 있다. 

가볍게 훑기에 좋다.

  


가볍게 훑기 좋은 또다른 자료 <http://www.slideshare.net/perone/deep-learning-convolutional-neural-networks> 도 참고하자. 이건 거의 CNN에 대한 자료다. 

  


  * Why deep?
    * deep할수록 같은 수의 파라메터로 더욱 non-linear한 학습을 할 수 있음
    * 혹은 같은 함수를 찾기 위해 더 적은 파라메터를 쓸 수 있음
  * CNN
    * Receptive Field (1959)
      * 전체를 다 보는게 아니라 지엽적으로 보면서 학습
    * Neocognitron (1980)
      * 단순한 구조와 복잡한 구조를 왔다갔다하면서 학습
    * LeNet-5 (LeCun, 1988)
      * 현재 CNN과 비슷한 구조
    * 이렇게 과거에도 이론적으론 비슷한 수준이었으나 컴퓨팅 파워의 문제 및 몇가지 아이디어의 부족으로 SVM에 밀림
    * CNNs in ILSVRC (ImageNet Large Scale Visual Recognition Challenge)
      * 120만장의 데이터를 주고 1000클래스에 대해 이미지 분류하는 대회
      * 여기서 엄청난 성능 향상을 보여줘서 빵 뜸
      * AlexNet (2012) - CNN RevoLUtion : 여기서 ReLU가 나옴! dropout도 여기서 나왔다는데...?
      * Goog-Le-Net (google, 2014) - Inception이라고 불리는 모듈을 많이 쌓아서 엄청 복잡함
      * 구르넷은 너무 복잡하니까 작은 convolution을 깊게 쌓는 방향으로 많이 한대
        * 마소가 22단계
        * 대신 parameter initialize를 잘 하는걸 고민
  *     * 요즘 비전쪽으론 거의 무조건 CNN
    * Components
      * Convolution
        * Receptive Fields 의 일종
        * 해당 Window (fields) 에 필터가 보고자 하는 shape이 있을때 반응한다는 개념
      * Pooling
        * 여러 필터들에 의한 Convolution 결과를 합침
        * 위치에 강건 (윈도우 안에서 어느 위치에 있던지 같은 액티베이션 값을 줌)
      * Activation function
        * add non-linearity
    * Visualizing - DeConvNet
      *   

      * ![](http://cfile21.uf.tistory.com/image/2139A0505637675222388B)
      *   

      * NN이 블랙박스인 문제를 해결하기 위해서, Reconstruction을 수행하여 각 레이어가 뭘 의미하는지를 파악함
      * pooling의 경우 복원이 불가능하므로 이를 해결하기 위해 중간중간 값을 저장하여 이 저장한 값을 이용하여 복원 (어느 위치가 max였는지)
    * Visualizing - Gradient Ascent
      *   

      * ![](http://cfile23.uf.tistory.com/image/247A824B5637676B1278D7)
      *   

      * 목표로 하는 activation 값을 정하고, 그에 맞게 input값을 변화시킴 (처음에는 랜덤 인풋). 그리고 그 activation 값을 증폭시킴
      * 해 보면 어떤 노드는 스쿨버스를 보고 어떤 노드는 플라밍고를 보고 이런걸 확인할 수 있대!
    * 그래서 CNN이 보는 세계와 사람이 보는 세계가 같은가?
      * 간단한 노이즈에도 이상하게 분류함
      * 아직 부족함 (구글 포토 - 흑인을 고릴라로 잘못 분류한 사례)
    * 이게 한계인가?
      * 그렇지 않다. 더 많은 데이터가 문제를 해결해 줄 것
      * Adversarial Traning
        * 하나는 페이크 데이터를 계속 만들고
        * 하나는 제대로 구분함
        * 이렇게 서로서로 도우면서 서로 강해지는게 adversarial traning!
  * RNN (Recurrent Neural Network)
    * 다양한 구조를 가질 수 있음
      *   

      * ![](http://cfile22.uf.tistory.com/image/257E454D5637678F0E6D98)
      *   

      * 이해는 잘 안가지만 그렇대
      * <http://karpathy.github.io/2015/05/21/rnn-effectiveness/> 요기에 나오는 그림임
    * Problem with RNNs
      * 기본적으로 매우 깊은 구조이기 때문에, Vanishing or exploding problem이 존재함 (대부분 vanishing. exploding은 빵 커져버리는거)
      * RNN을 쓰는 이유가 long-term dependency를 보고싶은 건데 볼 수가 없게 되버림
    * LSTM (Long-Short Term Memory)
      * 이를 해결하기 위해, vanish 자체를 없애고 히든노드간 weight를 1로 고정
      * 대신 input을 받아들일 것인가 말 것인가를 학습
      * 이는 사람의 기억도 옛날거라고 무조건 기억 안 나는게 아니라 오래됐어도 잘 기억나는게 있는것과도 일치함
      *   

      * ![](http://cfile8.uf.tistory.com/image/253AF14C563767CA197ABC)
      *   

      * 위 그림에 보면 셀프 커넥션은 1.0임!
      * 들어오고 나가는 걸 control!
    * Bi-directional LSTM
      * 앞의 컨텍스트만 중에한 게 아니라 뒤의 컨텍스트도 중요함
      * 앞뒤를 다 고려하는 방법
      * 타임 시퀀셜한 데이터 뿐만 아니라 Bioinformatics에서도 프로틴(단백질)같은 경우에 공간적으로 앞뒤가 있어서 여기에도 널리 활용됨
  * Recent Advances in Deep Learning
    * Deep Dream (2015, 구글)
      *   

      * ![](http://cfile2.uf.tistory.com/image/243F804C563767DF15AE2E)
      *   

      * 아까 노드에서 뭘 학습한지 보기 위해서 했던 것과 비슷. 단 이번에는 이미지를 넣음. 그러면 이 이미지에서 개를 봤던 노드의 activation값이 높았다면, 이 액티베이션 값을 더욱 증폭시키니까 이미지가 더욱 개처럼 된다.
      * 근데 이 결과가 개로 변함. 왜? 학습 이미지가 개가 많았대. 개를 보는 노드들이 가장 액티베이트 되어 있으니까 그렇게 됨.
    * Neural Style (2015, Gatys et al)
      *   

      * ![](http://cfile3.uf.tistory.com/image/24070049563767F616B64D)
      *   

      * 이미지를 'Style' 과 'Contents' 로 분리함
      * StyleNet?
      * 인풋 영상의 컨텐츠를 유지하면서 스타일을 "고흐"스타일에 일치시킴
      * 컨텐츠는 CNN의 feature map의 activation value로 결정되며
      * 스타일은 feature map을 vectorize(위치정보 삭제) 하고 correlation 값을 본다
      * <http://sanghyukchun.github.io/92/>참고
    * Char-RNN (A. Karpathy, 2015)
      * 캐릭터 레벨로 하나의 단어단어를 뱉어내는 RNN
      * Text Generator. 시드 단어에 맞는 텍스트를 생성함
      * Obama-RNN (Samin, 2015)
      * 유사하게 음악도 작곡할 수 있음 (B. L. Sturm, 2015)
    * Neural Translation (Sutskever, 2014)
    * Word Embedding (word2vec, doc2vec)
    * Embedding Multi-modal Data (R. Kiros et al, 2014)
      *   

      * ![](http://cfile27.uf.tistory.com/image/231F9B4C5637680B382FF1)
      *   

      * 이렇게 word embedding하는걸 이미지와 함께 할 수도 있음
    * Image Caption Generator (A. Karpathy, 2015)
      * Neural Talk
      * 이미지를 주면 이미지를 설명하는 caption 을 달아줌
      * 뉴럴넷의 특성 중 하난데 멀티모달 데이터를 같이 사용할 수 있음
      *   

      * ![](http://cfile30.uf.tistory.com/image/2738634E5637682604399B)
      *   

      * CNN + RNN
    * Attention Mechanism
      * 여기에서 그치지 않고 영상의 "어떤 부분"을 설명하는지를 매칭함
      * Training Networks "Where to See"
      * 번역이라고 하면 어떤 단어가 어떤 단어로 번역되었는지 등
      * <http://sanghyukchun.github.io/93/> 이건가?

  



[Tistory 원문보기](http://khanrc.tistory.com/126)
