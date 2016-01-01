---
layout: post
title: "TensorFlow - (2) Install"
tags: ['DataScience/Deep Learning']
date: 2015-12-03 01:36:00
---
# [TensorFlow](http://www.tensorflow.org/)

나중에 linux-gpu ver 도 설치할텐데 그때 추가.

## Install

### Mac

맥에서는 cpu버전밖에 지원을 하지 않는다.  
여러가지 설치법이 있으나, 나는 그냥 기본으로 시스템에 설치하기로 했다. virtualenv는 사실상 아카데믹한 환경에서는 별로 필요하지 않다. 

cpu버전은 홈페이지에 소개되어 있는 대로 그냥 깔면 되는데, 간단한 트러블슈팅이 있었다:

  * 아직까지 python 2만 지원한다. 이는 리눅스도 마찬가지로 텐서플로 자체가 파이썬 2만 지원함. 파이썬 3용도 개발중이라고 함.
  * 내 맥북의 경우 python 2와 python 3이 같이 깔려있다. 문제는 pip임. pip를 python 2용 pip와 python 3용 pip로 분리해야 한다. 
    * <http://stackoverflow.com/questions/33622613/tensorflow-installation-error-not-a-supported-wheel-on-this-platform>
    * pip3으로 텐서플로를 설치하려고 하면 위와 같이 wheel이 없다고 에러가 남. `get-pip.py`를 받아서 깔아주면 `pip2.7`을 사용할 수 있다. `pip2.7`이 제대로 설치되었는지는 `-V`를 통해 확인할 수 있음. 
    
    
    $ pip2.7 -V
    pip 1.5.6 from /Library/Python/2.7/site-packages/pip-1.5.6-py2.7.egg (python 2.7)
    


[Tistory 원문보기](http://khanrc.tistory.com/133)
