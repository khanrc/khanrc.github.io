---
layout: post
title: "Deep Learning Tutorial"
tags: ['DataScience/Deep Learning']
date: 2015-05-24 17:12:00
---
# [A Deep Learning Tutorial: From Perceptrons to Deep Networks](http://www.toptal.com/machine-learning/an-introduction-to-deep-learning-from-perceptrons-to-deep-networks)

위 링크의 요약번역.

딥러닝의 키 컨셉과 알고리즘에 대해 살펴본다. 여기선 다 뺐지만, 본문에서는 자바 코드도 소개하고 있다. 내가 잘 모르는 부분일수록 본문 그대로 번역하기 때문에 앞부분이 요약이 많고 뒤로 갈수록 본문 내용을 다 포함한다.

## Perceptrons: 초기 딥러닝 알고리즘

> 퍼셉트론이란 두뇌의 인지 능력을 모방하도록 만들어진 인위적인 네트워크를 의미한다

single perceptron은 하나의 linear classifier와 같다. 이걸론 부족하다.

## Feedforward Neural Networks for Deep Learning

Feedforward NN. 위 문제를 해결하기 위해 input에서 output으로 바로 가지 않고 중간에 hidden layer를 통해 non-linear문제를 해결한다.

![FNN](http://assets.toptal.io/uploads/blog/image/333/toptal-blog-image-1395721488746.png)

### Beyond Linearity

그러나 이것만으론 부족한 것이, 결국 linear들의 집합이기 때문에 activation function은 linear하다. 따라서 여기에 non-linear function을 붙여준다: 로지스틱, 탄젠트 등의 함수를.

### Training Perceptrons

일반적인 딥러닝의 supervised learning 알고리즘은 backpropagation이다. 이는, 간단하게, 트레이닝 샘플을 네트워크에 흘려보내서 결과를 얻고, 이 결과로 에러를 계산하여 네트워크의 가중치들을 재조정하는 방법이다.

에러는 다양하게 계산할 수 있으나 MSE(mean square error)가 일반적이고, 이 에러를 최소화하도록 가중치를 조정하는데 이 때 쓰이는 방법이 [stochastic gradient descent](http://en.wikipedia.org/wiki/Stochastic_gradient_descent)다. stochastic gradient descent, 즉 SGD는 그냥 번역하자면 확률적인 gradient descent로 매 이동시마다 전체 데이터셋을 다 사용하는 것이 아니라 일부만 뽑아서 사용한다. 이렇게 함으로써 퍼포먼스를 개선할 수 있고 local optima에 빠지는 문제도 개선할 수 있다.

이 gradient descent를 수행할 때, error의 미분을 구해야 하는데 이를 구하기 위해서는 output node부터 시작해서 거꾸로 진행해야 한다. 이 때문에 backpropagation이라 불린다.

### Hidden layer

The hidden layer is where the network stores it's internal abstract representation of the training data, similar to the way that a human brain (greatly simplified analogy) has an internal representation of the real world.

[universal approximation theorem](http://en.wikipedia.org/wiki/Universal_approximation_theorem)에 따르면, 하나의 히든 레이어로 모든 펑션을 표현할 수 있다. 그렇긴 하지만 딥러닝에서와 같이 실제로는 여러개의 히든 레이어를 사용하는 것이 더 좋은 결과를 보인다.

### The Problem with Large Networks

2개 이상의 히든 레이어를 사용할 때, 더 높은 레이어는 그 전 레이어에 한단계 더 높은 추상화를 더한다고 볼 수 있다. 위에서 언급했듯, 보통 더 많은 레이어를 사용하는 것이 더 좋은 결과를 보여주지만 여기에도 문제가 있다.

  1. [Vanishing gradients](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf): 히든 레이어가 늘어나게 되면, backpropagation에서 information을 전달할 때 낮은 레이어로 갈수록 점점 더 그 정보가 의미없게 된다. 다시 말해서, 낮은 레이어로 갈수록 gradient가 사라지게 되고 weights의 변화가 거의 없어진다.
  2. [Overfitting](http://en.wikipedia.org/wiki/Overfitting)

## Autoencoder

오토인코더는 전형적인 FNN인데, 데이터셋을 압축적이고 분배된 표현(인코딩) 으로 학습하는 것을 목표로 한다. 

> 아래를 읽어보면 알겠지만, 데이터를 그대로 딥러닝 하게 되면 오버피팅할 우려가 크므로, 이를 더 간단한 데이터로 인코딩하는 pre-processing 작업이다. PCA와 비슷한 면이 있을 것 같다.
> 
> 아래 그림에서 인풋 레이어와 히든 레이어에 +1이 있는데, 이건 왜 있는 지 잘 모르겠음.

![AUTOENCODER](http://assets.toptal.io/uploads/blog/image/340/toptal-blog-image-1395769623098.png)

개념적으로, 이 네트워크는 input을 다시 만들도록 트레이닝 된다. 즉, input과 target data가 같다. 다시 말해 output이 input과 같은 형태이지만 compressed 된다. 무슨 말인지 헷갈릴 수 있는데 아래 예시를 보자:

### Compressing the Input: Grayscale Images

트레이닝 데이터가 28x28의 그레이스케일 이미지로 구성되어 있다. 여기서 각 픽셀이 하나의 input neuron이라고 하자. 즉 총 784개의 인풋 뉴런이 존재하게 된다. 그러면 output layer에도 같은 수(784)의 유닛이 존재하게 되고 각 output unit의 target value도 그레이스케일 이미지의 각 픽셀에 대응하게 된다.

직관적으로, 이 구조는 트레이닝 데이터와 라벨을 사용하여 둘을 매핑하던 기존의 트레이닝과 다르게, 데이터 자체의 내부 구조와 특성을 트레이닝한다. (이러한 이유로 Autoencoder에서의 hidden layer는 feature detector라고도 불린다) 일반적으로 히든 유닛의 개수는 인풋/아웃풋 유닛의 개수보다 적으며, 이를 통해 몇몇 중요한 특성들만 뽑아내고 dimensionality reduction(차원 축소)을 수행할 수 있다.

궁극적으로, 인풋 데이터로부터 핵심 피처들을 뽑아내어 압축된 데이터를 표현할 수 있는 가운데의 hidden layer의 적은 수의 노드들을 학습시키고자 한다.

### Flu Illness

autoencoder에 대해서 더 자세히 보기 위해서, 하나의 예제를 더 살펴보자.

이 케이스에서는 감기 증상에 대한 간단한 데이터셋을 사용한다:

  * 6개의 바이너리 인풋
  * 처음 3개는 병의 증상. [1 0 0 0 0 0]은 고열, [0 1 0 0 0 0]은 기침 등
  * 마지막 3개는 "counter" 증상. 만약 환자가 이걸 갖고 있다면 병에 걸릴 확률이 감소한다. [0 0 0 1 0 0]은 감기 백신 등

이 데이터에서 처음 3개 중 2개 이상이 1이면 sick이고, 뒤 3개 중 2개 이상이 1이면 healthy라고 하자. 

이 상황에서, 6개의 input units과 6개의 output units, 그리고 2개의 hidden units을 두고 autoencoder를 트레이닝 한다. 그 결과로, "sick"를 나타내는 하나의 hidden unit과 "healthy"를 나타내는 또다른 hidden unit을 얻었다.

### Going Back to Machine Learning

근본적으로, 우리의 두 히든 유닛은 감기 증상 데이터셋의 컴팩트한 표현을 학습했다. 이를 통해 딥러닝 학습에서 overfitting을 피할 수 있다. 

## Restricted Boltzmann Machines

[RBM](http://en.wikipedia.org/wiki/Restricted_Boltzmann_machine). input data의 확률분포(probability distribution)을 학습할 수 있는 generative stochastic NN이다.

![RBM](http://assets.toptal.io/uploads/blog/image/351/toptal-blog-image-1395942212600.png)

RBM은 hidden, visible, bias 레이어로 구성되어 있다. FNN과는 다르게, visible - hidden 레이어간의 커넥션은 undirected다. 즉, 앞으로만 값이 전달되는 것이 아니라 양방향으로 서로 propagate될 수 있다. 또한, 마찬가지로 FNN과는 다르게, 이 네트워크는 fully connected이다. 이는 두 인접한 레이어간의 이야기인데, 이게 두 인접한 레이어가 아니라 모든 레이어간 fully connected인 경우 이를 두고 Boltzmann machine(BM) 이라 한다.

standard RBM은 binary hidden and visible units을 갖는다. 즉, 각 유닛은 0 또는 1로 활성화(activation)되고 이는 [베르누이 분포](http://en.wikipedia.org/wiki/Bernoulli_distribution)를 따른다. 단, 여기에는 다른 [non-linearity](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) 변수들이 존재한다.

RBM은 알려진 지 좀 되었지만, 최근에 이 네트워크를 트레이닝하는 알고리즘인 [contrastive divergence](http://deeplearning.net/tutorial/rbm.html#id1) unsupervised training algorithm이 소개되면서 다시 주목받고 있다.

### Contrastive Divergence

single-step contrastive divergence algorithm (CD-1)은 이렇게 작동한다:

  1. Positive phase:
    * input sample _v_가 input layer에 들어왔다고 하자
    * _v_는 FNN과 같이 hidden layer로 propagate된다. hidden layer가 _h_로 activation되었다고 하자
  2. Negative phase:
    * _h_는 visible layer로 역전파되어, 새로운 결과 _v'_을 만들어낸다
    * _v'_이 다시 hidden layer로 전파되어 _h'_을 생성한다
  3. Weight update:  
![update](http://assets.toptal.io/uploads/blog/image/350/toptal-blog-image-1395939562020.58.50_PM.png)  
이 때 _a_는 learning rate이며 _v, v', h, h', w_ 는 벡터다.

이 알고리즘은 positive phase에서 이 네트워크의 real world data에 대한 내부적인 representation을 반영한다(_h_). 그리고 negative phase에서는 이 내부적인 representation을 기반으로 하여 새로운 데이터 표현을 생성한다(_v'_). 이 데이터 생성의 최종 목적은 real world data와 가능한 한 가까워지는 것이고 이는 weight update 공식에 반영되었다.

다시 말하면, 이 네트워크는 어떻게 데이터가 표현될 수 있는지에 대한 perception을 가지고 있는 것이고, 따라서 이 perception에 따라 데이터를 재생산한다. 만약 이렇게 재생산한 데이터가 정확하지 않으면(real world와 맞지 않으면) 네트워크를 조정하고 다시 시도한다.

### Returning to the Flu

자 그럼 contrastive divergence를 자세히 살펴보기 위해 감기 문제로 돌아와 보자. RBM 네트워크는 6개의 visible unit과 2개의 hidden unit으로 구성된다. 이 네트워크를 contrastive divergence로 학습할 것이다. 감기 증상(symptom) _v_는 visible layer에 인풋 샘플로 들어간다. 트레이닝 동안, 이 증상은 visible layer에 다시 등장하고, 이 데이터는 hidden layer로 전파된다. hidden unit은 각각 sick/healthy state를 의미하는데 이는 autoencoder와 매우 유사하다.

수백번의 반복 이후에, 오토인코더와 유사한 결과를 확인할 수 있다: 하나의 히든 유닛은 샘플이 "sick"일 때 활성화되고, 다른 유닛은 샘플이 "healthy"일 때 활성화된다.

## Deep Networks

지금까지 autoencoder와 RBM이 효과적으로 특성을 추출한다는 것을 보았다. 그러나 이러한 특성들을 직접적으로 사용할 수 있는 경우는 거의 없다. 사실, 위의 데이터셋은 예외에 가깝다. 대신에, 우리는 간접적으로 이 추출한 특성들을 사용하는 방법이 필요하다.

다행히도, 이 구조들이 모여 [deep network](http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)를 형성한다는 것이 밝혀졌다. 이 네트워크는 기존의 backpropagation방법이 갖고 있는 _vanishing gradient_문제와 _overfitting_문제를 극복하기 위해 한번에 한 레이어씩 그리디하게 학습한다.

이 구조는 강력한 결과를 보여준다. 한 예로, 구글의 유명한 ["고양이" 논문](http://static.googleusercontent.com/media/research.google.com/en/us/pubs/archive/38115.pdf)이 있다. 이 논문은 unlabled data기반으로 사람과 고양이의 얼굴을 구분하기 위해 특별한 종류의 deep autoencoder를 사용하였다.

더 자세히 살펴보자.

### Stacked Autoencoders

이름에서 알 수 있는 것처럼, 이 네트워크는 여러 계층의(stacked)오토인코더로 구성되어 있다.

![stacked autoencoder](http://assets.toptal.io/uploads/blog/image/335/toptal-blog-image-1395721542588.png)

각 히든 레이어는 오토인코더이고, 오토인코더의 출력인 hidden layer는 다음 오토인코더의 input layer가 된다. greedy layer-wise 트레이닝 절차는 이렇다:

  1. 첫번째 오토인코더(빨간색 히든 1)를 트레이닝한다 - 모든 트레이닝 데이터와 backpropagation을 사용해서. 물론 위 네트워크에는 나오지 않지만 오토인코더 트레이닝 시에는 output layer 또한 필요하다.
  2. 두번째 오토인코더(초록색 히든 2)를 트레이닝한다. 첫번째 오토인코더의 hidden layer(1)가 input layer(2)가 되고, 필요하지 않은 output layer(2)는 지워버린다. (괄호 안 숫자는 몇번째 오토인코더인지를 의미한다) 첫번째 오토인코더에 데이터를 넣으면, 이것이 두번째 오토인코더의 output layer까지 전파된다(feed forward propagate). 이후 이 결과를 가지고 backpropagation하여 두번째 오토인코더의 가중치를 업데이트한다. 첫번째 오토인코더와 마찬가지로 모든 샘플을 사용한다.
  3. 이 과정을 모든 레이어에 걸쳐 수행한다. 
  4. 이 1-3과정을 _pre-training_이라 한다. 이 과정을 거치면 초기 가중치를 얻을 수 있다. 여기에는 input data와 output label간의 매핑이 없다. 일반적으로, 여기에 하나 또는 여러개의 fully connected layer를 끝에 덧붙이는 방식을 사용한다(위 네트워크에서 파란 output layer와 같이). 이제, 전체 네트워크는 backpropagation으로 학습한 multilayer perceptron이 되었다. 이 과정을 _fine-tuning_이라고 부르기도 한다.

stacked autoencoder는 네트워크의 가중치를 초기화하기 위한 효과적인 pre-training 메소드에 대한 모든것이다. 

> fine-tune(fine-tuning)은 아마 이 초기 네트워크 가중치 설정 이후에 실제로 트레이닝 하는 과정을 말하는 것 같다. 위와 같이 stacked autoencoder 뒤에 레이어를 더 붙여서 data-label mapping을 한다면 하나의 네트워크처럼 보이겠지만 unsupervised pre-training과 supervised fine-tuning이 결합된 것이다.

### Deep Belief Networks

autoencoder와 마찬가지로, 볼츠만 머신(BM)또한 쌓을(stack) 수 있다. 그것이 바로 _deep belief networks (DBNs)_ 이다.

![DBN](http://assets.toptal.io/uploads/blog/image/336/toptal-blog-image-1395721550849.png)

stacked autoencoder와 마찬가지로, _t_번째 RBM의 hidden layer는 다음 _t+1_번째 RBM의 visible layer가 된다. 첫번째 RBM의 input layer는 전체 네트워크의 input layer이며, 이 네트워크는 greedy layer-wise pre-training으로 학습된다:

  1. RBM(1)은 contrastive divergence와 모든 트레이닝 샘플로 학습된다
  2. RBM(2)는 hidden layer(1)을 visible layer(2)로 하여 학습된다. input sample이 visible layer(1)에 들어오면, hidden layer(1)로 propagate되고, RBM(2)를 학습시키는 데이터가 된다
  3. 위 과정을 모든 레이어에 대해 반복한다
  4. stacked autoencoder와 같이, pre-training 후에 뒤에 하나 이상의 fully connected layer를 붙여 fine tuning한다. 

이 과정은 stacked autoencoder와 유사하지만, autoencoder가 RBM으로 대체되고 backpropagation이 contrastive divergence algorithm으로 대체되었다.

## Convolutional Networks

마지막 딥러닝 아키텍처로, convolutional network를 살펴보자. 이미지 인식에 최적화된 feedforward network의 특수한 형태다.

> convolutional: 나선형의.

![CNN](http://deeplearning.net/tutorial/_images/mylenet.png)

CN의 실제 구조를 살펴보기 전에, 먼저 이미지 필터나 연관된 가중치로 이루어진 사각형을 정의하자. 이 필터는 전체 인풋 이미지에 대해 적용되고, 여러개의 필터를 적용할 수 있다. 예를 들어, 우리는 6x6 필터들을 인풋 이미지에 적용할 수 있다. 그러면, 아웃풋 픽셀[1,1]은 인풋 픽셀[1,1]과 6x6 필터의 가중합(weighted sum)이다. 

> 제대로 이해한 건지 잘 모르겠는데, 이해한 바로는 6x6필터가 있으면 각 픽셀들을 이 필터들에 다 적용시켜서, 가중합을 구해 출력한다는 의미인 듯하다. 6x6필터가 필터 36개가 아니라 필터 하나인 듯.

이 네트워크는 아래 특성들에 의해 정의된다:

  * **Convolutional layers**는 필터들을 인풋에 적용한다. 예를 들어, 위 예에서 첫번째 convolution layer는 4개의 6x6필터를 가질 수 있다. 이미지 전체에 적용된 한 필터의 결과는 _feature map_(FM)이라 불리고 이 피처 맵의 수는 필터의 수와 같다. 만약 이전의 레이어 또한 convolutional이라면, 그 레이어의 FM에 필터들이 적용된다. 즉, 각 input FM은 output FM으로 연결된다. 이미지의 가중치 공유는 위치에 상관없이 특성을 추출할 수 있게 하고, 다양한 필터의 사용은 다양한 특성 셋을 추출할 수 있게 한다.
  * **Subsampling layers**는 인풋의 사이즈를 줄인다. 예를 들어, 인풋이 32x32의 이미지로 구성되어 있고, 이 레이어가 2x2의 subsampling region을 갖고 있고, output value는 16x16의 image로 되어 있다고 하자. 이것은 각 2x2 인풋 픽셀이 하나(1x1)의 아웃풋 픽셀로 통합된다는 것을 의미한다. 서브샘플링 방법은 여러가지가 있는데, [max pooling](http://deeplearning.net/tutorial/lenet.html#maxpooling), [average pooling](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial), [stochastic pooling](http://techtalks.tv/talks/stochastic-pooling-for-regularization-of-deep-convolutional-neural-networks/58106/)이 대표적이다.
  * 마지막 레이어(subsampling or convolutional)는 일반적으로 하나 이상의 fully connected layer이고, 마지막 레이어가 타겟 데이터를 나타낸다.
  * 학습은 modified backpropagation으로 수행된다. 이 방법은 subsampling layer를 고려하고 필터가 적용된 모든 값들에 기반하여 convolutional filter weights를 업데이트한다.

[여기](http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)에서 자바스크립트로 만들어진 비슷한 네트워크의 시각화를 볼 수 있다.

> CNN은 잘 이해가 가지 않음. 따라서 번역도 이상할 수 있으니 본문을 참고하자.


[Tistory 원문보기](http://khanrc.tistory.com/92)
