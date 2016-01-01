---
layout: post
title: "Deep Learning을 위해 어떤 GPU를 써야 할까?"
tags: ['DataScience/Deep Learning']
date: 2015-11-19 20:45:00
---
## [Which GPU for Deep Learning?](http://timdettmers.com/2014/08/14/which-gpu-for-deep-learning/)

요약번역 + @

### 여러개의 GPU를 써야 할까?

  * 여러개의 GPU를 한 컴퓨터에서 사용함. 40Gbit/s 의 통신속도를 가짐.
  * DNN을 여러개의 GPU에서 분산처리하는게 매우 어려운데 비해 스피드업은 그다지 빠르지 않음
  * 단, Data parallel에서 그치지 않고 Model parallel을 통하면 성능향상을 볼 수 있음
  * 멀티 GPU의 사용은 복잡한 네트워크와 거대한 데이터가 있어야 의미가 있는데 대기업이 아닌이상 이정도의 데이터를 확보하기 힘듦
  * 하지만 멀티 GPU를 사용한다면 동시에 여러 알고리즘을 각각의 GPU에서 실험할 수 있다는 장점이 있음
  * 10-15GB 이하의 작은 데이터셋을 사용한다면, 싱글 GPU로 충분함
  * 단, CNN은 weight sharing 때문에 data parallel이 가능해서 멀티 GPU가 유용함. 그러나, 멀티 GPU를 사용하는 알고리즘을 코딩하는 것은 싱글 GPU에 비해 훨씬 어렵다는 것을 잊지 말자.
  * 결론적으로 매우 매우 큰 데이터셋을 사용하는게 아닌 한 싱글 GPU로 충분함

### 그래서 어떤 GPU를 써야 할까? NVIDIA or AMD?

  * 몰랐는데, GPGPU 라이브러리인 CUDA와 OpenCL은 각각 NVIDIA와 AMD에서 지원하는 기술임. 즉 NVIDIA 그래픽카드를 사면 CUDA를 써야 하고, AMD 그래픽카드를 사면 OpenCL을 써야함.
  * Deep learning 라이브러리들이 대부분 CUDA 기반이므로 NVIDIA를 사야함
  * 뿐만 아니라 CUDA 커뮤니티가 OpenCL에 비해 훨씬 훌륭함

### 간단한 NN을 위해 필요한 메모리 크기는?

  * 필자는 GTX Titan with 6GB memory를 사용함
  * 보통 이렇게까지 많이 필요하지 않음
  * dropout + [momentum/Nesterov(NAG)/AdaGrad/RMSProp](http://newsight.tistory.com/m/post/224) 를 사용한 DNN의 메모리 필요량을 아래와 같이 계산해 볼 수 있음:

> ![memory](http://s0.wp.com/latex.php?latex=%7B%5Cmbox%7BMemory+in+GB+%7D+%3D+12%5Ctimes+1024%5E%7B-3%7D%5Cleft%28%5Cleft%28%5Csum%5Climits_%7Bi%3D0%7D%5E%7B%5Cmbox%7Bweights%7D%7D+%5Cmbox%7Brows%7D_i%5Ctimes+%5Cmbox%7Bcolumns%7D_i+%5Cright%29+%2B+%5Cmbox%7Bbatchsize%7D%5Csum%5Climits_%7Bi%3D0%7D%5E%7B%5Cmbox%7Blayers%7D%7D+%5Cmbox%7Bunits%7D_i+%5Cright%29%7D&bg=ffffff&fg=000000&s=0)  
Memory formula: The units for the first layer is the dimensionality of the input. In words this formula means: Sum up the weight sizes and input sizes each; multiply the input sizes by the batch size; multiply everything by 4 for bytes and by another 3 for the momentum and gradient matrix for the first term, and the dropout and error matrix for the second term; divide to get gigabytes.

  * 이 공식을 통해 필자가 [Kaggle competition - Partly Sunny with a Chance of Hashtags](https://www.kaggle.com/c/crowdflower-weather-twitter)에서 사용한 [batchsize 128의 9000x4000x4000x32 크기의 NN](https://www.kaggle.com/c/crowdflower-weather-twitter/forums/t/6488/congratulations/35640#post35640) 의 메모리 사용량을 계산하면:  
![calculate](http://s0.wp.com/latex.php?latex=%7B12%5Ctimes+1024%5E%7B-3%7D+%28%289000%5Ctimes+4000+%2B+4000%5Ctimes+4000+%2B+4000+%5Ctimes+32%29+%2B+128%289000%2B4000%2B4000%2B32%29%29+%5Capprox+0.62%5Cmbox%7BGB%7D%7D+&bg=ffffff&fg=000000&s=0)  
이다. 즉, 1.5GB 정도의 메모리를 가진 작은 GPU에서도 잘 동작한다.
  * 단, CNN의 경우엔 이야기가 좀 달라짐

### 주어진 예산에서 가장 빠른 GPU를 찾아보자!

  * Memory bandwidth
    * GPU의 성능은 일반적으로 FLOPS (FLoating-point OPerations per Second) 라는 초당 부동소수점 연산으로 측정함
    * 그러나 우리에게 중요한 건 그게 아니라 bandwidth가 몇 GB/s 가 나오느냐임. 이는 메모리를 초당 몇번이나 읽고 쓸 수 있는지를 의미
    * dot product 등의 모든 수학 연산이 전부 이 memory read/write bandwidth에 의존함
    * [GPU bandwidths는 위키 페이지](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units#GeForce_700_Series) 를 참고하자
  * GPU(VGA) architecture
    * 또다른 중요한 룰은 그래픽카드의 아키텍처임
    * 900시리즈가 사용하는 Maxwell 아키텍처와 400, 500시리즈가 사용하는 Fermi 아키텍처가 600, 700시리즈가 사용하는 Kepler 아키텍처보다 빠름
    * 따라서 700(Kepler) 시리즈보다는 900(Maxwell) 시리즈를 고려해라.
  * 비싼 옵션: GTX Titan X or GTX 980
  * 저렴한 옵션: GTX 960 or GTX 680 with 4GB ram
  * RNN에 대해서는 GTX 970도 훌륭한 옵션이지만 복잡한 CNN을 돌릴 때는 3.5GB의 메모리와 괴상한 구조 때문에 문제가 생길 수 있음

### GTX 970

  * [GTX 970의 이상한 구조](http://hexus.net/tech/news/graphics/79925-nvidia-explains-geforce-gtx-970s-memory-problems/) 때문에 메모리를 3.5GB 이상 사용하면 문제가 생길 수 있음
  * 이는 large CNN 을 학습할 때 문제가 됨
  * 하지만 시끄러웠던 것에 비하면 딥러닝에서 큰 문제는 없다. 메모리를 3.75GB 이하로 사용한다면 GTX 960보다 빠름
  * 메모리를 3.5GB 이하로 관리할 수 있다면, 바꿔 말해 3.5GB 이상 사용할 때 그걸 경고해주는 라이브러리를 함께 사용한다면 GTX 970은 매우 효율적인 선택
  * 이런 트러블을 신경쓰고 싶지 않다면 4GB GTX 960 or GTX 680을 추천

### TL;DR advice

(그대로 퍼옴)

Best GPU overall: GTX Titan X  
Cost efficient but expensive: GTX Titan X, GTX 980, GTX 980 Ti  
Cost efficient but troubled: GTX 580 3GB (lacks software support) or GTX 970 (has memory problem)  
Cheapest card with no troubles: GTX 960 4GB or GTX 680  
I work with data sets &gt; 250GB: GTX Titan, GTX 980 Ti or GTX Titan X  
I have little money: GTX 680 3GB eBay  
I have almost no money: AWS GPU spot instance  
I do Kaggle: GTX 980 or GTX 960 4GB  
I am a researcher: 1-4x GTX Titan X  
I want to build a GPU cluster: This is really complicated, you can get some ideas [here](https://timdettmers.wordpress.com/2014/09/21/how-to-build-and-use-a-multi-gpu-system-for-deep-learning/)  
I started deep learning and I am serious about it: Start with one GTX 680, GTX 980, or GTX 970 and buy more of those as you feel the need for them; save money for Pascal GPUs in 2016 Q2/Q3 (they will be much faster than current GPUs)

### 결론

  * 돈이 충분하다면 GTX Titan X or GTX 980
  * 돈이 부족하다면 GTX 960 or GTX 680. 문제가 있지만 컨트롤할 수 있다면 GTX 970
  * 정말 돈이 없다면 AWS GPU instance.

### 참고

  * Tensorflow를 돌리려면 <https://developer.nvidia.com/cuda-gpus> 에서 3.5 이상이어야 함

### 기타

  * CNN 의 메모리 사이즈에 대한 단락은 생략했다. 본문을 참고하자. CNN은 메모리를 더 많이 필요로 하는 듯?
  * AWS GPU instance 에 대한 얘기도 생략했다. 저렴하게 사용하기에 좋으나 퍼포먼스가 부족하고 메모리 이슈 등이 존재.
  * 한 컴퓨터에서 멀티 GPU를 사용하는 것에 그치지 않고, [2대의 컴퓨터를 사용해서 총 4개의 GPU를 사용하는 포스트](http://timdettmers.com/2014/09/21/how-to-build-and-use-a-multi-gpu-system-for-deep-learning/)도 있다. GPU 클러스터를 만들고자 한다면 참고하자. 2대 이상의 컴퓨터를 사용할 때는 컴퓨터간 통신이 필요하고 따라서 Network bandwidth 이슈가 발생한다.


[Tistory 원문보기](http://khanrc.tistory.com/128)
