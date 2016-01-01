---
layout: post
title: "한글 in the dictionary (feat. pretty)"
tags: ['Python']
date: 2014-09-17 23:48:00
---
# 한글 in the dictionary (feat. pretty)

파이썬 2에서는 항상 한글이 문제다. 파이썬 3에선 괜찮다는데…  
json, 즉 딕셔너리에 한글을 담아서 찍으면 유니코드로 나온다.
    
    
    >>> dic = dict()
    >>> dic['key'] = "한글"
    >>> print dic
    {'key': '\xed\x95\x9c\xea\xb8\x80'}
    

## 읽을 수 있게 찍어보자

딕셔너리 오브젝트의 value를 직접 찍어주면 잘 나온다.
    
    
    >>> print dic['key']
    한글
    

근데 항상 이렇게 찍을 순 없지 않은가!

딕셔너리를 통째로 읽을 수 있게 찍어보자.
    
    
    >>> import json
    >>> print json.dumps(dic, ensure_ascii=False)
    {"key": "한글"}
    

성공!

## 예쁘게 찍어보자

위에서는 key가 하나였기에 문제가 없었지만 많아지면 보기 힘들어진다.
    
    
    >>> dic['k1'] = "k1"
    >>> dic['k2'] = "k2"
    >>> dic['k3'] = "k3"
    >>> print json.dumps(dic, ensure_ascii=False)
    {"k3": "k3", "k2": "k2", "k1": "k1", "key": "한글"}
    

으악!!!

pprint를 쓸 수도 있지만, 어차피 file wirte할 일도 있고 json에서 바로 예쁘게 찍을 수 있다.
    
    
    >>> print json.dumps(dic, ensure_ascii=False, indent=4)
    {
        "k3": "k3",
        "k2": "k2",
        "k1": "k1",
        "key": "한글"
    }
    

olleh!

json에서는 키 소팅 등 다양한 기능을 지원하니 필요하면 알아보도록 하자.

**참고**  
<https://docs.python.org/2/library/json.html>


[Tistory 원문보기](http://khanrc.tistory.com/36)
