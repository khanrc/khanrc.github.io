---
layout: post
title: "7 tips to Time Python scripts and control Memory & CPU usage"
tags: ['Python']
date: 2014-12-03 11:04:00
---
# [7 tips to Time Python scripts and control Memory &amp; CPU usage](http://www.marinamele.com/7-tips-to-time-python-scripts-and-control-memory-and-cpu-usage?utm_source=Python+Weekly+Newsletter&utm_campaign=2bbc6ad4bc-Python_Weekly_Issue_167_November_27_2014&utm_medium=email&utm_term=0_9e26887fc5-2bbc6ad4bc-312692397)

위 글의 번역.  
파이썬 스크립트의 시간을 측정하고, 메모리와 CPU 사용량을 컨트롤하는 7가지 팁.

실행하는데 오랜 시간이 걸리는 복잡한 파이썬 프로그램을 돌릴 때, 이 실행시간 즉 퍼포먼스를 향상시키고 싶어진다. 그럼 어떻게 할 것인가?

먼저, 코드의 바틀넥을 찾아내는 툴이 필요하다. 그래야 그 부분을 향상시키는데에 집중할 수 있다.

그리고 또한, 메모리와 CPU 사용량을 컨트롤해야한다 - 그것이 퍼포먼스를 향상시킬 새로운 방법을 제시할 것이다.

그러므로, 이 포스트에서는 함수의 실행시간과 메모리 및 CPU 사용량을 측정하고 향상시킬 수 있는 7가지 툴을 소개할 것이다.

## 1\. Use a decorator to time your functions

함수의 실행시간을 측정하는 데코레이터를 활용해라:
    
    
    import time
    from functools import wraps
    
    
    def fn_timer(function):
        @wraps(function)
        def function_timer(*args, **kwargs):
            t0 = time.time()
            result = function(*args, **kwargs)
            t1 = time.time()
            print ("Total time running %s: %s seconds" %
                   (function.func_name, str(t1-t0))
                   )
            return result
        return function_timer
    

이제 이렇게 쓸 수 있다:
    
    
    import random
    
    @fn_timer
    def random_sort(n):
        return sorted([random.random() for i in range(n)])
    
    
    if __name__ == "__main__":
        random_sort(2000000)
    
    
    # 결과:
    Total time running random_sort: 1.41124916077 seconds
    

## 2\. Using the timeit module

또다른 옵션은 `timeit`모듈을 사용하는 것이다. 이 모듈은 평균 시간을 측정해준다. 실행하기 위해서, 아래 커맨드를 터미널에서 쳐 보자:
    
    
    $ python -m timeit -n 4 -r 5 -s "import timing_functions" "timing_functions.random_sort(2000000)"
    ...
    4 loops, best of 5: 2.08 sec per loop
    

`timing_functions`는 우리가 만든 스크립트 이름이다. `-n 4`옵션으로 4번 실행해서 평균내고, `-r 5`옵션으로 각 테스트를 5번 반복하여 그중 best를 출력한다. 아래를 보면 확실히 알 수 있다:
    
    
    # 이렇게 실행하면 2번 실행해서 평균내고,
    $ python -m timeit -n 2 -r 1 -s "import timing_functions" "timing_functions.random_sort(2000000)"
    Total time running random_sort: 1.89807391167 seconds
    Total time running random_sort: 2.80085301399 seconds
    2 loops, best of 1: 2.42 sec per loop
    
    # 이렇게 실행하면 2번 실행해서 best를 보여준다.
    $ python -m timeit -n 1 -r 2 -s "import timing_functions" "timing_functions.random_sort(2000000)"
    Total time running random_sort: 1.94069004059 seconds
    Total time running random_sort: 2.66689682007 seconds
    1 loops, best of 2: 2.01 sec per loop
    
    # timeit 모듈이 위에서 wrapping하기 때문에 시간 측정시 약간 차이는 있다.
    

`-n`과 `-r`을 지정하지 않는다면 default로 10 loops와 5 repetitions를 돌게 된다.

## 3\. Using the time Unix command

그러나 이 `decorator`와 `timeit`모듈은 둘다 파이썬에 기반한다. 이렇게 파이썬에 의지하지 않고도 시간측정을 할 수 있다:
    
    
    $ time -p python timing_functions.py
    Total time running random_sort: 1.88052487373 seconds
    real 2.01
    user 1.80
    sys 0.20
    

출력의 첫줄은 데코레이터가 출력한 것이고, 두번째줄부터는:

  1. `real`은 이 스크립트를 실행하는데 걸린 총 시간을 의미한다.
  2. `user`는 이 스크립트를 실행하기 위해 CPU가 사용된 시간(CPU time spent)이다.
  3. `sys`는 kernel-level function에서 사용된 시간이다.

즉, 실제로 이 스크립트가 수행된 시간은 `user`시간이고 `sys`시간은 이 스크립트를 래핑하는 처리과정에 소모되는 시간이다. 그리고 `real` \- (`user` \+ `sys`) 는 I/O나 다른 태스크의 종료까지 기다리는데 걸린 시간이라고 할 수 있다.

## 4\. Using the cProfile module

만약 시간이 각각의 펑션과 메소드에서 얼마나 걸리는지 알고 싶다면, 그리고 각각이 몇번이나 불리는지 알고 싶다면 [`cProfile`](https://docs.python.org/2/library/profile.html) 모듈을 사용할 수 있다:
    
    
    $ python -m cProfile -s cumulative timing_functions.py
    
    Total time running random_sort: 4.75063300133 seconds
             2000067 function calls in 4.817 seconds
    
       Ordered by: cumulative time
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    0.065    0.065    4.817    4.817 timing_functions.py:1(<module>)
            1    0.000    0.000    4.751    4.751 timing_functions.py:7(function_timer)
            1    1.698    1.698    4.751    4.751 timing_functions.py:18(random_sort)
            1    1.535    1.535    1.535    1.535 {sorted}
      2000000    1.433    0.000    1.433    0.000 {method 'random' of '_random.Random' objects}
            1    0.084    0.084    0.084    0.084 {range}
            1    0.001    0.001    0.001    0.001 random.py:40(<module>)
            1    0.000    0.000    0.000    0.000 hashlib.py:55(<module>)
            6    0.000    0.000    0.000    0.000 hashlib.py:94(__get_openssl_constructor)
            1    0.000    0.000    0.000    0.000 random.py:91(__init__)
            1    0.000    0.000    0.000    0.000 {_hashlib.openssl_md5}
            1    0.000    0.000    0.000    0.000 random.py:100(seed)
            1    0.000    0.000    0.000    0.000 {math.exp}
            1    0.000    0.000    0.000    0.000 {posix.urandom}
            1    0.000    0.000    0.000    0.000 __future__.py:48(<module>)
            1    0.000    0.000    0.000    0.000 timing_functions.py:6(fn_timer)
            1    0.000    0.000    0.000    0.000 random.py:72(Random)
            1    0.000    0.000    0.000    0.000 functools.py:17(update_wrapper)
            2    0.000    0.000    0.000    0.000 {math.log}
            1    0.000    0.000    0.000    0.000 {function seed at 0x7f0a14976a28}
           11    0.000    0.000    0.000    0.000 {getattr}
            2    0.000    0.000    0.000    0.000 {time.time}
            7    0.000    0.000    0.000    0.000 __future__.py:75(__init__)
            1    0.000    0.000    0.000    0.000 {math.sqrt}
            1    0.000    0.000    0.000    0.000 {binascii.hexlify}
            6    0.000    0.000    0.000    0.000 {globals}
            1    0.000    0.000    0.000    0.000 functools.py:39(wraps)
            3    0.000    0.000    0.000    0.000 {setattr}
            1    0.000    0.000    0.000    0.000 {_hashlib.openssl_sha224}
            1    0.000    0.000    0.000    0.000 random.py:651(WichmannHill)
            1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
            1    0.000    0.000    0.000    0.000 {_hashlib.openssl_sha384}
            1    0.000    0.000    0.000    0.000 __future__.py:74(_Feature)
            1    0.000    0.000    0.000    0.000 {_hashlib.openssl_sha1}
            1    0.000    0.000    0.000    0.000 random.py:801(SystemRandom)
            1    0.000    0.000    0.000    0.000 {_hashlib.openssl_sha512}
            1    0.000    0.000    0.000    0.000 {_hashlib.openssl_sha256}
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
    

이렇게 각 펑션이 얼마나 불렸는지 디테일한 디스크립션을 볼 수 있다. 이 디스크립션은 누적시간(cumulative time spent)으로 정렬된다. 총 실행시간이 이전보다 높아진 걸 볼 수 있는데, 이는 위와 같이 디테일하게 측정하기 위해서 소모되는 측정시간이다.

## 5\. Using line_profiler module

[`line_profiler`](https://github.com/rkern/line_profiler)는 우리 스크립트의 각 라인에 대해 CPU time spent를 측정한다. 먼저, 인스톨부터 하자.
    
    
    $ sudo pip install line_profiler
    

이제, 스크립트에서 프로파일링을 하고 싶은 함수에 `@profile`데코레이터를 걸어주자. 어떤 import를 할 필요는 없다!
    
    
    @profile
    def random_sort2(n):
        l = [random.random() for i in range(n)]
        l.sort()
        return l
    
    if __name__ == "__main__":
        random_sort2(2000000)
    

이제 이 스크립트를 이렇게 실행시켜주자:
    
    
    $ kernprof -l -v timing_functions.py
    

`-l` 플래그는 line-by-line 분석을 의미하고, `-v`플래그는 verbose(장황한) output을 의미한다고 한다. `-l`을 빼면 `@profile`에서 에러가 나고, `-v`를 빼면 파일로 출력을 하는데 알 수 없는 인코딩이다. 즉, 그냥 둘 다 써주도록 하자 -.-;;

`@profile`을 빼고 `-l`을 빼면, 4번의 `cProfile` 모듈을 사용했을 때와 비슷한 결과가 나온다. 암튼 둘다 써줘야 원하는 결과를 얻을 수 있다 - 바로 이렇게:
    
    
    $ kernprof -l -v timing_functions.py
    Wrote profile results to timing_functions.py.lprof
    Timer unit: 1e-06 s
    
    Total time: 3.84659 s
    File: timing_functions.py
    Function: random_sort2 at line 22
    
    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
        22                                           @profile
        23                                           def random_sort2(n):
        24   2000001      2409084      1.2     62.6      l = [random.random() for i in range(n)]
        25         1      1437499 1437499.0     37.4      l.sort()
        26         1            4      4.0      0.0      return l
    

총 3.85초 중에서 랜덤 배열을 생성하는 데 62.6%의 시간이 소모되었으며 sort()함수는 37.4%의 시간을 소모한 것을 확인할 수 있다. 이 방법도 마찬가지로 시간 측정에 들어가는 오버헤드 때문에 스크립트 실행시간이 길어진다.

## 6\. Use the memory_profiler module

[`memory_profiler`](https://pypi.python.org/pypi/memory_profiler)모듈은 우리 스크립트의 메모리 사용량을 line-by-line으로 분석해준다. 그러나, 이 모듈은 프로그램을 더더욱 느리게 만든다.

먼저 두 모듈을 깔아주자:
    
    
    $ sudo pip install memory_profiler
    $ sudo pip install psutil
    

[`psutil`](https://github.com/giampaolo/psutil)은 `memory_profile`의 성능을 향상시키기 위해 설치한다. 그리고 분석한다:
    
    
    $ python -m memory_profiler timing_functions.py
    Total time running random_sort2: 275.303025007 seconds
    Filename: timing_functions.py
    
    Line #    Mem usage    Increment   Line Contents
    ================================================
        22    9.801 MiB    0.000 MiB   @fn_timer
        23                             @profile
        24                             def random_sort2(n):
        25  134.113 MiB  124.312 MiB       l = [random.random() for i in range(n)]
        26  126.359 MiB   -7.754 MiB       l.sort()
        27  126.359 MiB    0.000 MiB       return l
    

해 보면 알겠지만 어마어마하게 오래 걸린다 -_- 분명 `psutil`은 깔았는데… 처음에 그냥 했다가 하도 오래 걸려서 시간을 측정해보고자 @fn_timer까지 달아서 다시 돌렸다. 아무튼 끝나긴 하니까 계속 돌려보면 위와 같은 결과를 얻을 수 있다. 메모리 사용량이 `MiB`(mebibyte1)로 측정된다.

## 7\. Using the guppy package

마지막으로, 스크립트의 각 스테이지에서 어떤 오브젝트(str, tuple, dict …)들이 얼마나 생성되었는지를 트래킹 하고 싶을때 사용할 수 있는 패키지 `guppy`가 있다.
    
    
    $ pip install guppy
    

설치 후 아래와 같은 코드를 추가하자:
    
    
    from guppy import hpy
    
    
    def random_sort3(n):
        hp = hpy()
        print "Heap at the beginning of the function\n", hp.heap()
        l = [random.random() for i in range(n)]
        l.sort()
        print "Heap at the end of the function\n", hp.heap()
        return l
    
    
    if __name__ == "__main__":
        random_sort3(2000000)
    

그리고 이 코드를 실행시키면 아래와 같은 결과를 얻을 수 있다.
    
    
    $ python timing_functions.py
    Heap at the beginning of the function
    Partition of a set of 27118 objects. Total size = 3433904 bytes.
     Index  Count   %     Size   % Cumulative  % Kind (class / dict of class)
         0  12302  45   979136  29    979136  29 str
         1   6136  23   495528  14   1474664  43 tuple
         2    323   1   250568   7   1725232  50 dict (no owner)
         3     76   0   228640   7   1953872  57 dict of module
         4   1704   6   218112   6   2171984  63 types.CodeType
         5    206   1   217424   6   2389408  70 dict of type
         6   1646   6   197520   6   2586928  75 function
         7    206   1   183272   5   2770200  81 type
         8    125   0   136376   4   2906576  85 dict of class
         9   1050   4    84000   2   2990576  87 __builtin__.wrapper_descriptor
    <94 more rows. Type e.g. '_.more' to view.>
    Heap at the end of the function
    Partition of a set of 2027129 objects. Total size = 68213504 bytes.
     Index  Count   %     Size   % Cumulative  % Kind (class / dict of class)
         0 2000083  99 48001992  70  48001992  70 float
         1    181   0 16803240  25  64805232  95 list
         2  12304   1   979264   1  65784496  96 str
         3   6135   0   495464   1  66279960  97 tuple
         4    329   0   252248   0  66532208  98 dict (no owner)
         5     76   0   228640   0  66760848  98 dict of module
         6   1704   0   218112   0  66978960  98 types.CodeType
         7    206   0   217424   0  67196384  99 dict of type
         8   1645   0   197400   0  67393784  99 function
         9    206   0   183272   0  67577056  99 type
    <94 more rows. Type e.g. '_.more' to view.>
    

이렇게 원할 때 메모리의 heap 영역을 찍어볼 수 있다. 이를 통해서 우리 스크립트의 오브젝트 생성 및 삭제 플로우를 확인할 수 있다.

* * *

  1. mebibyte. MB(megabyte)가 종종 1024가 아니라 1000을 단위로 하기 때문에 문제가 된다. 이를 정확하게 1024를 단위로 한 것이 MiB이다. 즉 MiB가 MB의 Strict한 버전이라고 볼 수 있다. 마찬가지로 GiB, KiB 또한 존재한다. MiB ⊂ MB.↩


[Tistory 원문보기](http://khanrc.tistory.com/77)
