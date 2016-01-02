---
layout: post
title: "TensorFlow - (4) MNIST - Softmax Regression"
tags: ['Deep Learning']
date: 2015-12-03 01:39:00
---
# [TensorFlow](http://www.tensorflow.org/)

MNIST tutorial.

## [Softmax Regression](http://www.tensorflow.org/tutorials/mnist/beginners/index.html#softmax-regressions)

내가 softmax regression (multinomial logstic regression) 에 대해 자세히 몰라서 정리해본다. MNIST 데이터 사용. 

데이터의 클래스 라벨들에 대해 확률을 예측하고 싶다면 소프트맥스가 그것을 할 수 있다. 소프트맥스는 두 스텝으로 구성된다: evidence를 계산하고, 이를 probability로 변환하고. 

evidence를 계산하기 위해 픽셀 강도 (pixel intensities) 의 가중합 (weighted sum) 을 계산한다. 만약 높은 강도를 갖는 픽셀이 그 이미지에 불리한 증거 (evidence) 라면 weight는 음수가 되고, 그 픽셀이 유리한 증거라면 양수가 된다. 

> 자세히 이해는 안가지만, 클래스 라벨 즉 여기서는 0~9 이미지를 비교해서 0에서만 나타나는 픽셀 특징을 갖고 있으면 0에 대해 양수가 되고 그 반대면 음수가 되고 이런 식인듯. 

음수 가중치는 빨갛게 표시하고 양수 가중치는 파랗게 표시하면 다음과 같다:

![](http://cfile28.uf.tistory.com/image/246BE34056683EF80735D7)

  


또한 인풋 데이터에 독립적인 어떤 가중치가 있을 수 있다. 이를 bias라 한다. 최종적으로 인풋 $x$가 주어졌을 때 클래스 $i$에 대한 evidence는: 

<div>$$\text{evidence}_i = \sum_j W_{i,~ j} x_j + b_i$$</div>

자 이제 이렇게 계산한 evidence를 softmax 함수를 통해 확률로 변환할 수 있다. 

<div>$$y = \text{softmax}(\text{evidence})$$</div>

softmax는 "activation" 혹은 "link" 를 제공한다. linear 한 값 (여기서는 evidence) 을 우리가 원하는 형태 (여기서는 10가지 클래스 라벨에 대한 확률분포) 로 변환해준다. 즉 evidence를 확률로 변환하는 과정이라고 이해할 수 있다. 이는 다음과 같이 정의된다: 

<div>$$\text{softmax}(x) = \text{normalize}(\exp(x))$$</div>

식을 풀면 이렇게 된다: 

<div>$$\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$</div>

그러나 첫 식으로 이해하는게 더 도움이 될 것이다 - exp(input) 을 normalize 하는 것. exp는 결과의 차이를 부각시키는 효과를 갖고, normalize는 결과를 확률분포로 변환하기 위함이다.

> 원문에는 위 식에 대한 설명이 주저리 적혀 있는데, 내가 보기엔 매우 심플하게 softmax는 간극을 벌리고 그 결과를 확률화한다. 즉, 클래스 라벨에 대해 각각 evidence를 계산한 다음, 이 evidence가 큰 클래스 라벨의 확률을 그 evidence 이상으로 크게 만들어 주는 것. 어찌보면 당연할 수 있는데, 10개의 클래스 중 9개의 evidence가 1이고 1개만 2라고 한다면 그 클래스가 정답일 확률은 2/11 보다는 훨씬 클 것이다.

이 과정을 다이어그램으로 나타내면 다음과 같다.

  


![](http://cfile9.uf.tistory.com/image/267D3C4556683F121791D2)

  


이는 결국 이렇게 쓸 수 있다: 

<div>$$y = \text{softmax}(Wx + b)$$</div>

## [Deep MNIST for Experts](http://www.tensorflow.org/tutorials/mnist/pros/index.html)

위 링크에서 Softmax Regression 부분.

### Load MNIST Data

MNIST 데이터셋은 60000개의 트레이닝 데이터와 10000개의 테스트 데이터로 구성된다. 각 이미지는 28x28x1 (0/1의 흑백 채널) 이며 0~9의 class label을 갖는다.
    
    
    # coding=utf-8
    import tensorflow as tf
    import input_data
    
    
    # download data
    # mnist is traning/validation/test set as Numpy array
    # Also it provides a function for iterating through data minibatches
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    

### Start TensorFlow InteractiveSession
    
    
    # InteractiveSession을 쓰지 않으면 세션에 그래프를 올리기 전에 그래프를 전부 완성해야 함.
    # 다른말로 하면 계산하기 전에 그래프를 완성해야 함. InteractiveSession을 쓰면 그때그때 계산이 가능함.
    sess = tf.InteractiveSession()
    

#### Computation graph

파이썬에서 연산을 빠르게 하기 위해 numpy와 같은 라이브러리들은 파이썬 바깥 (아마도 C) 에서 연산을 수행하게 한다.  
그러나 이런 노력에도 불구하고 이러한 switching 자체가 오버헤드임. 특히 GPU나 분산처리할 때 심각함.  
이러한 문제를 해결하기 위해 텐서플로는 연산 하나하나를 C로 수행하는게 아니라 연산 전체를 통째로 C로 수행함.  
이러한 접근방식은 Theano나 Torch와 유사  
파이썬의 역할은 전체 컴퓨테이션 그래프를 구축하고, 어느 부분이 실행되어야 하는지를 명시하는 일이다. 

즉 그래프를 만들고 run()을 통해 원하는 부분을 실행하면 해당 부분이 Session에 올려져 실행이 되는데 이 작업이 통째로 외부에서 실행되는듯.  
그렇기 때문에 원래는 InteractiveSession을 쓰면 안 되는거임. InteractiveSession을 쓰면 속도가 상당히 느려지리라 짐작함.

본문에서는 파이썬 바깥 혹은 파이썬으로부터 독립적으로 연산한다고 표현하므로 정확히 C인지는 잘 모르겠음. 참고로 여기서 C 라고 하면 C++도 포함.

### Softmax Regression Model

#### Placeholder
    
    
    # 1-layer NN => Softmax. 1-layer란 Input layer와 Output layer만 있는 것이 1-layer임.
    # placeholder는 실행할때 우리가 값을 넣을 수 있음
    x = tf.placeholder("float", shape=[None, 784])  # x는 input image. 784 = 28*28, 이미지를 핌 (flatten). 흑백 이미지이므로 각 픽셀은 0/1
    y_ = tf.placeholder("float", shape=[None, 10])  # _y는 class label. mnist가 0~9까지의 이미지이므로 10개. one-hot 벡터.
    

#### Variables
    
    
    # Variables: Weights & Bias
    # Variable은 말 그대로 변수에 해당한다.
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    # 변수는 사용하기 전에 초기화해줘야 한다. 선언시에 정해준 값으로 (여기서는 0) 초기화된다.
    sess.run(tf.initialize_all_variables())  # 모든 변수 초기화
    

#### Predicted Class and Cost Function
    
    
    # softmax 함수는 이미 구현되어 있으므로 한줄에 짤 수 있음.
    # tf.nn.softmax는 소프트맥스 함수만을 말하고 이 과정 전체가 소프트맥스 리그레션이다.
    y = tf.nn.softmax(tf.matmul(x, W) + b) # Wx+b = Output nodes의 액티베이션 값. 즉 액티베이션 값을 소프트맥스 함수에 넣음.
    
    # Cost function: 트레이닝 과정에서 최소화해야 하는 값.
    # cross-entropy between the target and the model's prediction.
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    # reduce_sum은 텐서의 합을 구함. reduce는 텐서를 축소한다는 개념인 듯.
    """
    # Example)
    # 'x' is [[1, 1, 1]]
    #         [1, 1, 1]]
    tf.reduce_sum(x) ==> 6
    tf.reduce_sum(x, 0) ==> [2, 2, 2]
    tf.reduce_sum(x, 1) ==> [3, 3]
    tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
    tf.reduce_sum(x, [0, 1]) ==> 6
    """
    

### Train the Model
    
    
    # 지금까지, 우리의 모델과 코스트 펑션을 정의했음.
    # 즉 다시 말해서 텐서플로는 우리의 전체 컴퓨테이션 그래프를 알고 있음.
    # 이에 기반하여 자동으로 미분하여 (differentiation) gradient를 계산할 수 있음.
    # 텐서플로는 다양한 최적화 알고리즘을 제공함. http://www.tensorflow.org/api_docs/python/train.html#optimizers
    # 여기서는 steepest gradient descent를 사용.
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    # 이 한 줄은 그래프에 하나의 op를 추가한 것. 이 op는 gradient를 계산하고, 파라메터 업데이트 스텝을 계산하고, 그 결과를 파라메터에 적용한다.
    # 즉 위 스텝을 반복하면 점점 학습이 됨.
    
    # 50 크기의 mini-batch로 1000번 학습을 함.
    for i in range(1000):
        batch = mnist.train.next_batch(50) # mini-batch (50)
        # batch[0]은 x, batch[1]은 y로 구성됨. [data, class_label] 구조.
        # batch[0 or 1][0~49] 가 각각의 데이터.
        # batch_xs, batch_ys = mnist.train.next_batch(50) 형태로 받는 것이 더 직관적.
        train_step.run(feed_dict={x: batch[0], y_:batch[1]})
        y_eval = y.eval(feed_dict={x:batch[0]})
    

### Evaluate the Model
    
    
    # argmax는 tensor에서 max값을 찾는 함수다. 파라메터는 tensor, axis (dimension) 임. axis개념은 numpy와도 같고 위의 reduce_sum과도 같음.
    # reduce_mean도 마찬가지.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  # 예측한 라벨과 정답 라벨을 비교하고
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 비교한 결과의 평균을 낸다
    print accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels})  # 실행
    
    # => 0.9092
    
    # 참고: Tensor.eval(), Operation.run(). (in InteractiveSession)
    

### More

더 자세한 건 아래 문서들을 참고하자.  
<http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/>  
<http://neuralnetworksanddeeplearning.com/chap3.html#softmax> (원문 추천) 


[Tistory 원문보기](http://khanrc.tistory.com/135)
