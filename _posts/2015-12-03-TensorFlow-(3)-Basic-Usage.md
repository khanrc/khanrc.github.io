---
layout: post
title: "TensorFlow - (3) Basic Usage"
tags: ['DataScience/Deep Learning']
date: 2015-12-03 01:38:00
---
# [TensorFlow](http://www.tensorflow.org/)

## Basic Usage

텐서플로를 쓰기 위해서는 먼저 이해해야 한다.

  * 컴퓨테이션을 그래프로 나타낸다
  * 그래프는 `Sessions` 위에서 실행한다
  * 데이터는 tensor로 나타낸다
  * 상태를 `Variables` 와 함께 유지한다
  * 데이터를 쓰기 (연산하기) 위해 feeds와 fetches를 사용한다

지금은 이해가 잘 안 되는데, 다 읽고 나면 이해가 됨.

### Overview

텐서플로는 컴퓨테이션을 그래프로 나타내는 시스템이다. 노드는 op (operation) 라고 불린다. op는 0개 이상의 `Tensor`를 받고, 0개 이상의 텐서를 생산한다. 텐서는 다차원의 어레이 (multi-dimensional array) 다. 예를 들어, 이미지의 mini-batch를 [batch, height, width, channels]의 4D floating point array로 표현할 수 있다.

> gradient descent 를 계산할 때, full-batch 라고 하면 전체 데이터를 다 사용해서 gradient를 구하는 것이고, mini-batch는 일부분만을 사용하고 on-line 은 한 샘플만을 사용한다. 즉 full-batch는 기본 gradient descent이고 mini-batch와 online은 SGD (stochastic gradient descent) 이다. 참고로 full-batch, mini-batch, online learning 은 이것을 말하는 것이다.

텐서플로 그래프는 컴퓨테이션의 설명이다. 무언가를 컴퓨팅하기 위해서는, 그래프가 `Session`에 올려져야 (launch) 한다. 세션은 그래프의 ops (operations) 를 `Devices` (CPU or GPU) 에 올리고, op를 실행하기 위한 메소드를 제공한다. 그러면 이 메소드는 ops에 의해 생성된 텐서를 리턴하는데 이는 `numpy의 ndarray` 이고 `C and C++의 tensorflow::Tensor` 이다.

### The computation graph

텐서플로는 그래프를 조립하는 construction phase와 그래프의 ops를 실행하기 위해 세션을 사용하는 execution phase로 구성된다. 

예를 들면, NN을 학습하는 그래프는 construction phase에서 생성되고, 이것이 반복적으로 실행되면서 실질적으로 트레이닝 ops가 실행되는 것은 execution phase에서 일어난다.

텐서플로는 C, C++, Python에서 사용할 수 있지만 그래프를 조립하는 작업은 파이썬에서 훨씬 쉽게 가능함. 반면 세션 라이브러리는 세 언어에서 동일하게 사용가능함.

#### Building the graph
    
    
    import tensorflow as tf
    
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    
    product = tf.matmul(matrix1, matrix2)
    

이렇게 쓰면, 바로 계산되는 것이 아니라, ops constructor가 default graph에 nodes를 추가한다. 대부분의 경우에 이 디폴트 그래프로 전체 플로우를 나타낼 수 있다. 여러개의 그래프가 필요할 경우 [Graph class](http://www.tensorflow.org/api_docs/python/framework.html#Graph) 도큐먼트를 참조하자.

이제, 디폴트 그래프는 3개의 노드를 가진다: 2개의 constant() ops 와 하나의 matmul() op. 실제로 결과를 얻기 위해서는 이 그래프를 세션에 올려야 한다.

#### Launching the graph in a session
    
    
    sess = tf.Session()
    
    # The output of the op is returned in 'result' as a numpy `ndarray` object.
    result = sess.run(product)
    print result
    # ==> [[ 12.]]
    
    sess.close()
    

세션에서 모든 연산은 패러렐하게 작동한다. 세션 constructor에 파라메터를 넘기지 않았기 때문에 디폴트 그래프로 작동한다. 세션 API는 [Session class](http://www.tensorflow.org/api_docs/python/client.html#session-management) 도큐먼트를 참고하자.

작업이 끝나면 리소스를 해방하기 위해 close()가 필요하다. 파이썬에서 이러한 구조는 `with` 를 통해 간단하게 쓸 수 있다:
    
    
    with tf.Session() as sess:
      result = sess.run(product)
      print result
    

텐서플로는 그래프를 패러렐하게 작동시킨다. 이는 자동으로 작동하지만 원한다면 명시적으로 작동시킬수도 있다.
    
    
    with tf.Session() as sess:
      with tf.device("/gpu:1"):
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.],[2.]])
        product = tf.matmul(matrix1, matrix2)
        ...
    

위 예제는 두번째 gpu를 작동시킨다. 물론 있을 때 얘기. [Using GPU](http://www.tensorflow.org/how_tos/using_gpu/index.html) 참고.

### Interactive Usage

이와 같이, 텐서플로에는 그래프를 만들고 이를 세션에 올려서 `Session.run()` 을 통해 실행시킨다. 이러한 과정은 `IPython` 등의 interactive python environment에서는 불편할 수 있으므로 이를 위한 환경을 제공한다 - `InteractiveSession`, `Tensor.eval()`, `Operation.run()`
    
    
    import tensorflow as tf
    sess = tf.InteractiveSession()
    
    x = tf.Variable([1.0, 2.0])
    a = tf.constant([3.0, 3.0])
    
    x.initializer.run()
    
    sub = tf.sub(x, a)
    print sub.eval()
    # ==> [-2. -1.]
    

자세히는 모르겠지만 Variable로 선언된 x는 initializer를 거쳐야 초기화가 되는 듯 하며, sess변수는 어디에서도 쓰이지 않지만 InteractiveSession() 을 생성하지 않으면 에러가 난다. run() 과 eval() 에서 자동적으로 참조가 되는 듯.

### Tensors

텐서플로에서 모든 데이터는 텐서로 나타낸다. 컴퓨테이션 그래프에서 노드 (오퍼레이션) 들을 이동할 수 있는 것은 오직 텐서뿐이다. 텐서는 n차원의 어레이나 리스트로 생각할 수 있다. 텐서는 고정된 (static) type, rank, shape을 가지고 있다. [Rank, Shape, and Type](http://www.tensorflow.org/resources/dims_types.html) 참조. Rank는 차원 수를 의미한다.

### Variables

변수는 그래프의 실행 사이에 상태를 유지한다. [Variables](http://www.tensorflow.org/how_tos/variables/index.html) 참고. 간단한 카운터 예제를 보자:
    
    
    # Create a Variable, that will be initialized to the scalar value 0.
    state = tf.Variable(0, name="counter")
    # Variable을 생성하는 것 또한 노드 (op) 다.
    
    # Create an Op to add one to `state`.
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)
    # 이 작업은 그래프에서 3개의 노드로 구성된다. 이후 sess.run(update) 가 되면 재귀적으로 세 단계가 실행된다.
    
    # Variables must be initialized by running an `init` Op after having launched the graph.  We first have to add the `init` Op to the graph.
    init_op = tf.initialize_all_variables()
    # 변수 (Variable) 는 `init` Op 를 통해 초기화되어야 한다. 즉 변수를 쓰기 이전에 `init`를 실행시켜야 하고 그러려면 일단 그래프에 올려야 한다.
    
    # Launch the graph and run the ops.
    with tf.Session() as sess:
      # Run the 'init' op
      sess.run(init_op)
      # Print the initial value of 'state'
      print sess.run(state)
      # sess.run(state)는 state 변수를 생성하는 생성노드를 실행시킨다.
    
      # Run the op that updates 'state' and print 'state'.
      for _ in range(3):
        sess.run(update)
        # sess.run(update)는 state를 update하는 노드를 실행시킨ㄴ다.
        print sess.run(state)
    
    # output:
    
    # 0
    # 1
    # 2
    # 3
    

개인적으로 느끼기에 텐서플로의 활용은 함수형 언어 그리고 memoization (메모리 펑션) 을 닮았다. 연산 과정이 각각의 연산 (op) 들로 쪼개지고, 이 연산들이 각각 노드가 된다. 결과를 나타내는 마지막 노드를 실행시키면 그 노드는 차례차례 필요한 노드를 호출하여 재귀적으로 모든 연산을 실행시킨다. 예를 들면 위 과정에서 sess.run(update)는 tf.assign(state, new_value) 라는 op인데, 이 op는 다시 new_value 노드를 호출하고 이는 tf.add(state, one) 이라는 op이다. 그리고 이는 다시 one 노드로 가서 tf.constant(1) 이라는 op를 호출할 것이다. 반면, state는 Variable이기 때문에 초기 생성자 tf.Variable(0, name="counter") 까지 올라가지 않고 state에 저장된 값을 사용한다.

### Fetches

연산의 결과를 fetch 하는 (가져오는) 함수는, 이미 위에서도 많이 썼지만, `run()`이다. 지금까지는 싱글 노드만을 fetch했지만 여러개도 할 수 있다.
    
    
    input1 = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)
    intermed = tf.add(input2, input3)
    mul = tf.mul(input1, intermed)
    
    with tf.Session() as sess:
      result = sess.run([mul, intermed])
      print result
    
    # output:
    # [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]
    

All the ops needed to produce the values of the requested tensors are run once (not once per requested tensor).  
영어 문장이 100% 이해가 안 되는데, 같은 인풋 (텐서) 에 대해서 op는 한번만 계산한다는 의미인 듯 하다.

### Feeds

fetch가 값을 가져오는 개념이었다면, feed는 값을 넣는 개념이다. 지금까지 값을 넣기 위해 `Constant`와 `Variable`을 사용했다. 이 외에 텐서를 직접적으로 그래프의 op에 넣는 방법도 있다.
    
    
    # input1 = tf.placeholder(tf.types.float32)
    # input1 = tf.constant(1) # error! 
    input1 = tf.constant(1.)
    input2 = tf.placeholder(tf.types.float32)
    output = tf.mul(input1, input2)
    
    with tf.Session() as sess:
      print sess.run([output], feed_dict={input1:[7.], input2:[2.]})
    
    # output:
    # [array([ 14.], dtype=float32)]
    

위와 같이 run()에서 feed를 넣어주면 넣은 tensor가 임시로 op (위 예제에서는 tf.placeholder()) 를 대체한다. 일반적으로 "feed" 를 하기 위해서 명시적으로 op를 생성하고자 할 때 tf.placeholder()를 사용한다. 위 예제 맨 위의 주석과 같이, constant와 같이 다른 op 를 대체할 수 있으나 type은 같아야 한다 (tf.constant(1) 을 하면 에러가 난다).

placeholder()의 경우 feed를 넣지 않으면 에러가 난다.

[MNIST fully-connected feed tutorial](http://www.tensorflow.org/tutorials/mnist/tf/index.html) ([source code](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mnist/fully_connected_feed.py)) 는 larger-scale feed example이다.


[Tistory 원문보기](http://khanrc.tistory.com/134)
