---
layout: post
title: "Autoencoder vs RBM (+ vs CNN)"
tags: ['DataScience/Deep Learning']
date: 2015-11-29 02:11:00
---
# Autoencoder vs RBM (+ vs CNN)

<http://stats.stackexchange.com/questions/114385/what-is-the-difference-between-convolutional-neural-networks-restricted-boltzma>

딥러닝은 크게 unsupervised pretraining 페이즈와 supervised fine-tuning 페이즈로 나눌 수 있다. 여기서 대표적인 pretraining 모델이 오토인코더와 RBM이다. 

오토인코더와 RBM은 둘 다 unsupervised pretraining 기법이다. 즉, 클래스 라벨로부터 backpropagation 하는 것이 아니라 인풋을 아웃풋으로 하여 인풋을 reconstruction 할 수 있는 hidden units를 찾아내는 과정이다. 이 히든 유닛들은 데이터로부터 뽑아낸 feature라고 할 수 있으며 이 과정을 data-specific kernel이라고도 할 수 있다. 

차이가 있다면 RBM은 에너지 모델을 차용하고 그 에너지 모델은 볼츠만 분포 (boltzmann distribution) 에 기반하므로 이는 generative model이다. 반면, 오토인코더는 deterministic (discriminant) 모델이다. 

오토인코더는 피처를 축소하는 심플한 개념이고, RBM은 확률분포에 기반하여 visible 변수들과 hidden 변수들간에 어떤 상관관계가 있고 어떻게 상호작용하는지를 파악하는 개념이다. 즉, 인풋 (visible) 변수와 히든 변수의 joint probability distribution을 학습하고자 하는 것이다.

오토인코더가 직관적일 뿐 아니라 구현하기도 더 쉽고 파라메터가 더 적어서 튜닝하기가 쉽다. 대신 RBM은 generative 모델이기 때문에 오토인코더와는 달리 찾아낸 확률분포로부터 새로운 데이터를 생성할 수 있다. 파라메터가 많은만큼 더욱 유연하다.

CNN 경우에도 결국 유사한 개념이나, spatial한 개념이 들어간다. CNN도 오토인코더처럼 discriminant한 함수를 학습한다. 

이 글의 주가 되는 위 링크에서는 CNN을 unsupervised pretraining인 오토인코더나 RBM과 같은 맥락에서 취급하지만, 실제로 CNN은 대부분 directly supervised training이다. 이는, Stacked RBM과 Stacked autoencoder가 각각 2006년, 2007년에 소개되었는데, vanishing gradient 문제를 해결한 ReLU가 2009년에 등장하면서 그리고 데이터의 양이 증가하면서 점차 unsupervised pretraining의 중요성이 감소하였고, CNN은 1989년부터 있던 개념이지만 deep structure는 2012년에 등장하였는데 이 시점에서는 이미 충분한 데이터가 확보되어 거의 대부분의 CNN이 pretraining 없이 바로 supervised learning을 통해 학습하게 되었다. 물론, 다양한 CNN이 존재하고 개중에는 supervised pretraining을 하는 CNN도 있다. [Pre-training in deep convolutional neural network?](http://stats.stackexchange.com/questions/163600/pre-training-in-deep-convolutional-neural-network) 참고.

### Dimensionality reduction

차원축소에서 가장 유명한 개념은 PCA다. PCA는 "component" 라고 불리는 "internal axis" 들을 찾아내고, 그 중 중요한 몇가지만을 추출함으로써 차원축소를 수행한다. 오토인코터와 RBM도 동일한 개념이다. 잘 학습하고 나면, 데이터에서 노이즈를 제거하고, 히든 유닛은 오브젝트의 어떤 중요한 숨겨진 피처를 학습하게 된다 - 영화의 장르라던가, 이미지에서 눈썹의 모양이라던가. 

### Deep architecture

그렇다면 PCA와의 차이점은 무엇인가? PCA는 오직 데이터의 **linear transformation** 만이 가능하다. 아무리 PCA를 여러번 적용해도 마찬가지다. 반면, 오토인코더와 RBM은 non-linear한 특성을 학습할 수 있고 따라서 더욱 복잡한 관계를 학습할 수 있다. 더욱이 이것들은 **stacked** 될 수 있고, 쌓일수록 더욱 강력해진다. 

![face recog](http://i.stack.imgur.com/oGBRR.jpg)

위와 같이 레이어가 쌓일수록 더욱 추상화된 피처를 학습한다. 위로 갈수록 더 큰 공간상의 피처를 나타내는데, 이는 위에서 사용한 딥러닝 모델이 공간적인 특성을 사용하는 CNN이기 때문이다.

### Classification

위에서 언급한 모든 딥러닝 모델들은 전부 classification으로 직접적으로 쓰이진 않는다. 대신, low-level representation을 high-level로 변환하는 방법을 학습하기 위한 **pretraining**으로 사용한다. 그렇게 프리트레이닝을 통해 추상화된 하이레벨 피처를 학습한 이후 그 피처를 기반으로 SVM이나 logistic regression (softmax) 등의 classifier를 학습한다. 위의 이미지 예시의 경우에도 제일 아래에 최종적으로 학습한 피처로 classification을 수행하는 classification layer (혹은 component) 가 하나 추가되어야 한다. pretraining 이후 최종적으로 classifier를 학습하는 과정을 **fine-tuning**이라 한다. 이 때 최종적으로 추가된 classifier layer만이 학습되는 것이 아니라 전체 네트워크가 조정된다 (그런 것 같다).

### Furthermore

  * [Autoencoder와 RBM은 여전히 많이 쓰이는가?](https://www.quora.com/Are-RBM-autoencoders-still-in-use-in-current-deep-learning-projects)
    * 그렇다. 다만 CNN과 RNN이 워낙 핫해서 묻히는것 뿐 
  * [RBM vs Autoencoder](https://www.quora.com/When-should-I-use-an-autoencoder-versus-an-RBM)
    * 성능은 그때그때 다름. 또한 두개를 같이 쓸 수도 있음 - [Temporal Autoencoding RBM](http://arxiv.org/pdf/1210.8353.pdf)
  * [What is the fine-tuning](http://metaoptimize.com/qa/questions/10918/what-exactly-is-the-fine-tuning-phase-in-deep-learning)
    * fine-tuning은 최종 classification layer를 학습하면서 그 에러를 backpropagation하여 pretraining에서 구축한 deep network도 튜닝한다. 즉, fine-tuning 단계에서 마지막의 classification layer만 학습하는 것이 아니다.


[Tistory 원문보기](http://khanrc.tistory.com/131)
