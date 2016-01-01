---
layout: post
title: "String option - u와 r"
tags: ['Python']
date: 2014-09-21 19:46:00
---
# String option - u와 r

<http://stackoverflow.com/questions/2081640/what-exactly-do-u-and-r-string-flags-do-in-python-and-what-are-raw-string-l>

그냥 str : 8-bits ascii  
`u` : unicode  
`r` : raw string

`r`은 escape character로 사용되는 백슬래쉬 \를 문자 그대로 쓰일 수 있게 해준다.  
`ur`도 사용할 수 있는데, 마찬가지로 유니코드에서 \를 문자 그대로 쓸 수 있게 해줌.

## Test
    
    
    >>> s1 = "안녕\n방가"
    >>> s2 = r"안녕\n방가"
    >>> s3 = u"안녕\n방가"
    >>> s4 = ur"안녕\n방가"
    >>> s1
    '\xec\x95\x88\xeb\x85\x95\n\xeb\xb0\xa9\xea\xb0\x80'
    >>> s2
    '\xec\x95\x88\xeb\x85\x95\\n\xeb\xb0\xa9\xea\xb0\x80'
    >>> s3
    u'\uc548\ub155\n\ubc29\uac00'
    >>> s4
    u'\uc548\ub155\\n\ubc29\uac00'
    

네 문자열은 전부 다르다. 찬찬히 살펴보면 그 차이를 알 수 있다.


[Tistory 원문보기](http://khanrc.tistory.com/37)
