---
layout: post
title: "PyCon: 위대한 dict 이해하고 사용하기"
tags: ['Python']
date: 2014-11-10 02:08:00
---
# [위대한 dict 이해하고 사용하기](http://www.pycon.kr/2014/program/2)

파이콘 2014에서 발표되었던 내용이다. 좋은 내용이 많은 것 같아서 몇 가지만 정리한다. 슬라이드가 잘 되어 있으니 슬라이드를 살펴보길 추천한다. 동영상도 있고.

## basic usage

`dict`의 기본적인 활용예

### comprehension
    
    
    >>> a
    {'a': 1, 'c': 3, 'b': 2}
    >>> {v: k for k, v in a.items()}
    {1: 'a', 2: 'b', 3: 'c'}
    

위와 같이 dict comprehension은 { }로 한다. list는 [ ]다.

### get()

`dict`에서 key-value확인은 몇가지 방법이 있다.
    
    
    >>> a['a']
    1
    >>> a['d']
    KeyError: 'd'
    >>> 'a' in a
    True
    >>> 'd' in a
    False
    >>> a.get('a')
    1
    >>> a.get('d')
    None
    >>> a.get('d', 0)
    0
    

key가 존재하는지 `in`으로 확인할 수 있다. value를 가져오려면 dict[key]를 쓰던가, dict.get(key) 로 가져올 수 있다. dict[key]는 key가 없으면 에러가 나고, get은 None을 리턴한다. 위에서 보듯 없을때의 리턴값을 정해줄 수 있다.

이를 활용하면 아래와 같은 게 가능하다.
    
    
    # a['d'] += 1 이 하고 싶은데, 'd'가 있는지 없는지 모른다면?
    
    >>> a['d'] = a.get('d', 0) + 1
    >>> a
    {'a': 1, 'c': 3, 'b': 2, 'd': 1}
    

dict[key] += 1은 상당히 자주 사용한다.

### setdefault()

말 그대로 default를 세팅하는 함수다.
    
    
    {'a': 1, 'c': 3, 'b': 2}
    >>> a.setdefault('d', 0)
    0
    >>> a.setdefault('a', 0)
    1
    >>> a
    {'a': 1, 'c': 3, 'b': 2, 'd': 0}
    

key가 없으면 value에 default를 넣고, value를 리턴한다. 이를 이용해서도 dict[key] += 1문제를 해결할 수 있다. get()과 비슷한 것 같지만 슬라이드를 보면 다르게 활용하는 예가 나온다. 참고하자.

### 삭제
    
    
    >>> del a['d']
    >>> a
    {'a': 1, 'c': 3, 'b': 2}
    

### 순회
    
    
    >>> a.keys()
    ['a', 'c', 'b']
    >>> [k for k in a.keys()]
    ['a', 'c', 'b']
    >>> a.values()
    [1, 3, 2]
    >>> a.items()
    [('a', 1), ('c', 3), ('b', 2)]
    

iterkeys(), itervalues(), iteritems() 도 있다. 아마 range()와 xrange()의 차이일 것이다.

## understand

`dict`는 hash 기반으로 작동한다. 이건 뻔하지만, 슬라이드를 보면 더 깊숙히 들어간다. 여유가 있다면 동영상을 보도록 하자. 아무튼 결론은, `dict`는 (거의) O(1)이다.

## dictlike classes

`dict`의 확장인지 뭔지 아무튼 dict스러운 클래스들이 있다.

  * collections.OrderedDict
  * collections.defaultDict
  * collections.Counter
  * shelve.Shelf

### OrderedDict
    
    
    >>> from collections import OrderedDict
    >>> d = OrderedDict()
    >>> d['a'] = 1
    >>> d['b'] = 2
    >>> d['c'] = 3
    >>> d['d'] = 4
    >>> d.items()
    [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
    

순서가 있는 `dict`. 

### defaultDict
    
    
    >>> from collections import defaultdict
    >>> d = defaultdict(lambda: [])
    >>> d['a'].append(1)
    >>> d['a']
    [1]
    >>> d['b']
    []
    

`dict`의 default값을 지정해줄 수 있다. setdefault()를 dict 전체에 적용한 것. 이를 활용하면 2차원 `dict`도 쉽게 만들 수 있고 그 이상도 가능하다.
    
    
    >>> a = defaultdict(lambda: defaultdict(list))
    >>> a['hi']['hoi'] = 'hey'
    >>> a
    defaultdict(<function <lambda> at 0x10058da28>, {'hi': defaultdict(<type 'list'>, {'hoi': 'hey'})})
    

이렇게 신박한 것도 가능하다!
    
    
    >>> infinite_dict = lambda: defaultdict(infinite_dict)
    >>> inf = infinite_dict()
    >>> inf['a'] = 1
    >>> inf['b']['c'] = 2
    >>> inf['c']['d']['e'] = 3
    >>> inf
    defaultdict(<function <lambda> at 0x10058daa0>, {'a': 1, 'c': defaultdict(<function <lambda> at 0x10058daa0>, {'d': defaultdict(<function <lambda> at 0x10058daa0>, {'e': 3})}), 'b': defaultdict(<function <lambda> at 0x10058daa0>, {'c': 2})})
    

신박신박.

### shelve.Shelf

`shelve`는 iOS의 UserDefaults같은 기능이다. 데이터를 쉽게 디스크에 저장하여 보존할 수 있도록 도와준다.
    
    
    >>> from shelve import open
    >>> shelf = open('test') # test.db에 dict를 기록한다
    >>> shelf['hello'] = 1
    >>> shelf['hi'] = 2
    >>> shelf[1] = 3
    TypeError: dbm mappings have string indices only
    >>> shelf.close()
    

  * close() 를 해야 한다: `with`와, context manager인 `contextlib.closing`을 사용하자.
  * 문자열 키만 가능하다: 위에서도 확인할 수 있다.
  * test.db에 저장된다: 실제로 test.db가 생겨 거기에 저장된다.


[Tistory 원문보기](http://khanrc.tistory.com/71)
