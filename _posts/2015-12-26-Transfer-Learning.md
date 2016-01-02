---
layout: post
title: "Transfer Learning"
tags: ['Deep Learning']
date: 2015-12-26 22:42:00
---
# [Transfer Learning](http://cs231n.github.io/transfer-learning/)

트랜스퍼 러닝이란 딥러닝을 feature extractor로만 사용하고 그렇게 추출한 피처를 가지고 다른 모델을 학습하는 것을 말한다. 여기서는 (위 링크에서는) CNN 에서의 Transfer learning 에 대해 설명한다.

실제로 CNN을 구축하는 경우 대부분 처음부터 (random initialization) 학습하지는 않는다. 대신에, ImageNet과 같은 대형 데이터셋을 사용해서 pretrain된 ConvNet을 이용한다. 크게 3가지의 시나리오가 있다:

> 원문에 3가지라고 나와서 그대로 옮김. 앞쪽 레이어만 가지고 fine-tuning 하는 걸 따로 세서 3가지 인 듯.

  * **ConvNet as fixed feature extractor**. pretrained CNN 에서 마지막 classification layer (보통 softmax layer) 만 제거하면 완전한 feature extractor다. 이렇게 추출한 피처들을 **CNN codes** 라고 부른다. 알렉스넷 (AlexNet) 의 경우 이 CNN codes 는 4096-D codes다. 이 피처들을 사용해서, 우리의 training set은 linear classifier (e.g. Linear SVM or Softmax) 를 학습하기 위해 사용한다.
  * **Fine-tuning the ConvNet**. 마지막 classification layer만을 retrain하는 것이 아니라 pretrain된 전체 네트워크를 재조정 (fine-tuning) 하는 것. 상대적으로 사용할 수 있는 데이터가 많을 때 적합하다. 경우에 따라서 앞쪽 레이어 (앞쪽 레이어일수록 더욱 general한 feature를 학습하므로) 는 고정시키고 뒤쪽 레이어만 fine-tuning 하기도 한다. 

**Pretrained models**. 최근의 CNN은 여러개의 GPU를 써도 학습하는 데에 몇주씩 걸린다. 이렇게 학습된 CNN의 weights들이 공유되므로 그걸 사용하자.

**When and how to fine-tune?** 우리의 새로운 데이터가 큰지/작은지, 새 데이터가 pretrain에 사용된 원래 데이터와 비슷한지/다른지에 따라 4가지 시나리오가 있다. 

  1. 새 데이터가 작지만 원래 데이터와 비슷한 경우: CNN codes를 사용해서 linear classifier을 학습.
  2. 새 데이터가 크고 원래 데이터와 비슷한 경우: fine-tuning through the full network.
  3. 새 데이터가 작고 원래 데이터와 매우 다른 경우: 데이터가 작으므로 1번처럼 CNN codes를 사용해서 linear classifier를 학습해야겠지만, 문제는 원래 데이터와 달라서 그러면 안 됨. 대신에 네트워크의 앞쪽 레이어의 activation 값을 사용해서 SVM을 학습하자.
  4. 새 데이터가 크고 원래 데이터와 매우 다른 경우: 데이터가 크므로, 처음부터 CNN을 구축해도 되겠지만, 이러한 경우에도 pretrained model을 사용하는 것이 더 좋다. 데이터가 충분하므로 전체 네트워크를 fine-tuning 하자.


[Tistory 원문보기](http://khanrc.tistory.com/139)
