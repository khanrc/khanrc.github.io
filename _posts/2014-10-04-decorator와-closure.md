---
layout: post
title: "decorator와 closure"
tags: ['Python']
date: 2014-10-04 17:21:00
---
# decorator

데코레이터는 파이썬의 강력한 문법 중 하나다. 파이썬에 입문해서 이것저것 좀 하다 보면 여기저기서 많이 볼 수 있다. 데코레이터를 한 마디로 정리하자면, 함수를 래핑하여 앞뒤에서 전처리와 후처리를 하는 기능 이라고 할 수 있다. 파이썬에서는 이 데코레이터 기능을 간편하게 지원한다.

글이 쓸데없이 장황해졌는데, 윗 부분은 개념 설명이고 **in python**부터 보아도 무방하다.

## function: first class object

파이썬에서, 함수는 **first class 오브젝트**다. 다시 말해 변수와 함께 동등한 레벨의 객체로 취급된다. 따라서 우리는 자유자재로 변수처럼 함수를 인자로 넘길 수 있다. 대표적으로 `sorted`함수를 사용할 때를 생각해보자.
    
    
    >>> student_tuples = [
            ('john', 'A', 15),
            ('jane', 'B', 12),
            ('dave', 'B', 10),
    ]
    >>> sorted(student_tuples, key=lambda student: student[2])   # sort by age
    [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
    

## closure

파이썬은 **function closure**를 지원한다.
    
    
    >>> def outer():
    ...     x = 1
    ...     def inner():
    ...         print x # 1
    ...     return inner
    >>> foo = outer()
    >>> foo()
    1
    >>> foo.func_closure
    (<cell at 0x...: int object at 0x...>,)
    

즉 이런 경우다. foo는 inner를 리턴받았고, inner에서 사용하는 x는 inner 바깥의 outer에 있기 때문에 foo를 호출하는 시점에서는 x가 존재하지 않아야 한다. 그런데, 위에서 보이듯이 잘 실행된다. 이게 바로 **function closure**다. 펑션 클로저는 그 함수가 정의될 때 자신을 감싸고 있는 namespace가 어떻게 생겼는지 기억한다는 의미다. foo의 `func_closure`로 그 함수를 감싸고 있는 scope의 변수들을 볼 수 있다.

간단하게 말하면, 어떠한 함수를 객체로 받을 때 그 함수를 감싸는 scope의 변수들 또한 같이 가져간다는 의미다. 따라서 이러한 것도 가능하다:
    
    
    >>> def outer(x):
    ...     def inner():
    ...         print x # 1
    ...     return inner
    >>> print1 = outer(1)
    >>> print2 = outer(2)
    >>> print1()
    1
    >>> print2()
    2
    

## decorator

데코레이터는, 결론부터 말하자면, 이름 그대로 함수를 데코레이트 해준다. 함수를 인자로 받아 꾸며주는 기능을 지원한다. 바꿔 말하면 함수를 래핑하는 기능이라고도 할 수 있겠다.
    
    
    def verbose(func):
        def new_func():
            print "Begin", func.__name__
            func()
            print "End", func.__name__
        return new_func
    
    def my_function():
        print "hello, world."
    
    >>> my_function = verbose(my_function)
    >>> my_function()
    Begin my_function
    hello, world.
    End my_function
    

verbose라는 데코레이터를 통해 my_function을 데코레이트했다. 원래는 hello, world만 출력하는 함수였지만 이젠 데코레이트되어 앞뒤로 시작과 끝을 출력한다.

### in python

사실 지금까지는 이론이라고 할 수 있고, 이제부터가 진짜 코드 레벨이다. 파이썬 데코레이터라고 하면 바로 `@`를 의미한다. 파이썬 2.4에서 추가되었다고 한다. 파이썬 코드를 보다보면 아래와 같은 코드들을 종종 볼 수 있다.
    
    
    @verbose
    def my_function():
        print "hello, world."
    

이는 verbose라는 데코레이터로 my_function이라는 함수를 데코레이트 해준다는 것을 의미한다. 이 함수를 실행하면 같은 결과를 볼 수 있다.
    
    
    >>> my_function()
    Begin my_function
    hello, world.
    End my_function
    

헌데, my_function에 파라메터가 있으면 어떡하지? 위에서 했던 걸 생각해보면, 그냥 파라메터를 넣어 주면 된다.
    
    
    def verbose(func):
        def new_func(name):
            print "Begin", func.__name__
            func(name)
            print "End", func.__name__
        return new_func
    
    @verbose
    def my_function(name):
        print "hello,", name
    
    >>> my_function("hi")
    Begin my_function
    hello, hi
    End my_function
    

### [*args, **kwargs](http://codeflow.co.kr/question/311/args-%EC%99%80-kwargs-%EA%B0%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80%EC%9A%94/)

그런데 이렇게 되면 파라메터가 하나 있는 함수에 대해서만 이 데코레이터를 사용할 수 있다. 함수의 시작과 끝을 알리는 데 파라메터의 개수가 무슨 상관이란 말인가? 이런 쓸데없는 제약을 없애기 위해 사용할 수 있는 것이 바로 `*args`와 `**kwargs`다. 둘 다 지정되지 않는 파라메터들을 받지만, `**kwargs`는 딕셔너리로서 이름이 지정된 파라메터들을 받는다.
    
    
    >>> def foo(x, *args, **kwargs):
    ...     print x
    ...     print args
    ...     print kwargs
    >>> foo(1, 2, 3, 4, 5, y=2, z=3)
    1
    (2, 3, 4, 5)
    {'y': 2, 'z': 3}
    

이를 이용하면 데코레이터를 최종적으로 확장할 수 있다.
    
    
    def verbose(func):
        def new_func(*args, **kwargs):
            print "Begin", func.__name__
            func(*args, **kwargs)
            print "End", func.__name__
        return new_func
    

### class

다른 함수들과 마찬가지로, 데코레이터 함수도 클래스로 구현할 수 있다.
    
    
    class Verbose:
        def __init__(self, f):
            print "Initializing Verbose"
            self.func = f
    
        def __call__(self, *args, **kwargs):
            print "Begin", self.func.__name__
            self.func(*args, **kwargs)
            print "End", self.func.__name__
    
    
    @Verbose
    def my_function(name):
        print "hello,", name
    

실행 해 보면 결과는 동일하게 나온다.

## 참고

[파이썬 데코레이터 이해하기](http://codeflow.co.kr/question/731/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%8D%B0%EC%BD%94%EB%A0%88%EC%9D%B4%ED%84%B0-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0/) : 이론 위주 설명  
[파이썬 데코레이터 (decorator): 기초편](http://trowind.tistory.com/72) : 코드레벨 설명


[Tistory 원문보기](http://khanrc.tistory.com/47)
