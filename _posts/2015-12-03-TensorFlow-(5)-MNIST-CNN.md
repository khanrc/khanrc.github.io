---
layout: post
title: "TensorFlow - (5) MNIST - CNN"
tags: ['Deep Learning']
date: 2015-12-03 01:40:00
---
# [TensorFlow](http://www.tensorflow.org/)

MNIST tutorial.

## [Deep MNIST for Experts](http://www.tensorflow.org/tutorials/mnist/pros/index.html)

(4) MNIST - Softmax Regression 의 결과인 91%의 정확도는 충분하지 않다. 약간 수정해서 작은 CNN을 만들어 보자. state-of-art는 당연히 아니지만, 99.2%의 정확도가 나온다.

### Weight Initialization

모델을 만들기 위해, 많은 weight와 bias를 만들어야 할 필요가 있다. gradient가 0이 나오는 평형 상태를 방지하고 평형 상태를 깨기 위해 (symmetry breaking) 적은 양의 노이즈로 초기화해야 한다. 우리는 ReLU를 쓸 것이므로, "dead neuron" 을 피하기 위해 약간 양의 초기 bias로 초기화하는 것도 괜찮다. 이를 하기 전에 두가지 편리한 함수를 만들어두자.

> symmetry breaking 이란 물리학 개념으로 작은 떨림이 시스템을 결정하는 것이라고 함
    
    
    # truncated normal distribution에 기반해서 랜덤한 값으로 초기화
    def weight_variable(shape):
        # tf.truncated_normal:
        # Outputs random values from a truncated normal distribution.
        # values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    # 0.1로 초기화
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    

### Convolution and Pooling

여기서 사용하는 옵션들은 전부 기초적인 CNN임. convolution과 pooling에서도 마찬가지.
    
    
    # convolution & max pooling
    # vanila version of CNN
    # x (아래 함수들에서) : A 4-D `Tensor` with shape `[batch, height, width, channels]`
    def conv2d(x, W):
        # stride = 1, zero padding은 input과 output의 size가 같도록.
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    

### First Convolutional Layer
    
    
    # [5, 5, 1, 32]: 5x5 convolution patch, 1 input channel, 32 output channel.
    # MNIST의 pixel은 0/1로 표현되는 1개의 벡터이므로 1 input channel임.
    # CIFAR-10 같이 color인 경우에는 RGB 3개의 벡터로 표현되므로 3 input channel일 것이다.
    # Shape을 아래와 같이 넣으면 넣은 그대로 5x5x1x32의 텐서를 생성함.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # 최종적으로, 32개의 output channel에 대해 각각 5x5의 convolution patch (filter) weight 와 1개의 bias 를 갖게 됨.
    
    # x는 [None, 784] (위 placeholder에서 선언). 이건 [batch, 28*28] 이다.
    # x_image는 [batch, 28, 28, 1] 이 됨. -1은 batch size를 유지하는 것이고 1은 color channel.
    x_image = tf.reshape(x, [-1,28,28,1])
    
    # 이제, x_image를 weight tensor와 convolve하고 bias를 더한 뒤 ReLU를 적용하자. 그리고 마지막으론 max pooling.
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    

### Second Convolutional Layer
    
    
    # channels (features) : 32 => 64
    # 5x5x32x64 짜리 weights.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    

### Densely Connected Layer

2x2의 max pooling을 두번 거쳤으니 이제 우리가 갖고있는 이미지의 크기는 7x7이다. 이제 전체 이미지에 연결된 1024개의 뉴런으로 구성된 fully-connected layer를 추가하자. 이전 레이어, 즉 풀링 레이어2 (h_pool2) 의 텐서를 batch of vector로 변환하고, weight matrix를 곱하고, bias를 더하고, ReLU를 적용하자.
    
    
    # Densely connected layer
    # 7*7*64는 h_pool2의 output (7*7의 reduced image * 64개의 채널). 1024는 fc layer의 뉴런 수.
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # -1은 batch size를 유지하는 것.
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    

#### Dropout

오버피팅을 최소화하기 위해 readout layer 이전에 dropout을 적용하자. 텐서플로의 `tf.nn.dropout`은 자동으로 scaling하고 masking하므로 추가적인 작업이 필요 없다.

> scaling과 masking은 드롭아웃이 노드를 선택해서 제거하고 (masking) 그로 인해 전체 노드의 수가 줄어드는 것 (scaling) 을 의미하는 듯.
    
    
    # keen_prob은 dropout을 적용할지 말지에 대한 확률임. 이를 이용해서 training 동안만 드롭아웃을 적용하고 testing 때는 적용하지 않는다.
    # training & evaluation 코드를 보니 keen_prob = 1.0일때 dropout off 인 듯.
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    

### Readout Layer

Readout layer는 그냥 소프트맥스 레이어임. 그냥 softmax 적용하면 됨.
    
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    

### Train and Evaluate the Model

트레이닝과 평가는 Softmax Regression에서 했던 것과 동일하다. 단, optimizer를 steepest gradient descent 대신에 ADAM 을 사용한다. 또한 dropout 확률인 keen_prob가 feed_dict에 추가된다.
    
    
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g" % (i, train_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    print "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    

20000번을 돌리면 최종적으로 99.2%가 나온다고 한다. 거기까진 못 돌려봤고 Softmax regression을 돌린 횟수인 1000번 까지의 결과는 다음과 같다. 참고로, 소프트맥스 리그레션을 20000번 돌려도 정확도는 92%다.
    
    
    step 0, training accuracy 0.12
    step 100, training accuracy 0.8
    step 200, training accuracy 0.92
    step 300, training accuracy 0.88
    step 400, training accuracy 0.96
    step 500, training accuracy 0.9
    step 600, training accuracy 1
    step 700, training accuracy 0.96
    step 800, training accuracy 0.86
    step 900, training accuracy 1
    step 1000, training accuracy 0.96
    


[Tistory 원문보기](http://khanrc.tistory.com/136)
