---
layout: post
title: "Errors and Exceptions"
tags: ['Python']
date: 2014-10-09 11:14:00
---
# python: Errors and Exceptions

자바를 제대로 공부한 적이 없어서, try-catch 형태의 패턴에 익숙하지 않다. 물론 try catch가 자바에서만 쓰는 건 아니지만 지금까지 다른 언어로 개발하면서 딱히 필요성을 느낀 적이 없었다. 에러가 나면 고치면 되는 일이고.

헌데 파이썬에선 많이 쓰는, 써야 하는 것으로 보인다.

## try-except

파이썬에서는 try except 형태다. `try`에서 에러가 나면, `except`에서 받는다.
    
    
    try:
        ret = db.users.insert({
            '_id': 1,
            'name': 'cjb'
            })
    
        print "성공? ", ret
    except pymongo.errors.DuplicateKeyError, e:
        print "중복키 에러", e
    except:
        print "원인을 알수 없셤 : " + str(sys.exc_info())
    

`mongodb`에 `insert`를 하고 에러가 나면 그에 따른 처리를 하는 코드다. 첫번째 `except`와 같이 에러를 명시해 주면 해당 에러만 받아들이며, 뒤에 에러 변수를 지정해 주면 에러 내용을 받아볼 수 있다[^1].

두번째 `except`는 그 외의 모든 에러를 받는데, 위와 같이 `sys.exc_info()`를 출력하여 에러 내용을 볼 수 있다.

### raise

`except`가 있다면 `throw`도 있어야 인지상정. 파이썬에서는 `raise`라 한다.
    
    
    def test_raise():
        raise NameError("whynot")
    
    if __name__ == "__main__":
        try:
            test_raise()
    
            print "성공? ", ret
        except NameError, e:
            print "네임에러", e
        except:
            print "원인을 알수 없셤 : " + str(sys.exc_info())
    

이렇게 에러를 명시적으로 발생시킬 수 있다. 위와 같이 하면 e에 "whynot"이 들어간다.

[^1]: python 2 기준. python 3에서는 `, e` 대신에 `as e` 라고 쓴다.

### Exception as e

이 글을 쓰고 난 지 한참이 지나서야 Exception으로 받을 수 있다는걸 알게 되었다 -_-;;  
참고로 지금까지의 소스는 python 2 기준이었지만 아래는 python 3이다.
    
    
    import sys
    
    def test_raise():
        raise NameError("whynot")
    
    if __name__ == "__main__":
        try:
            test_raise()
        except Exception as e:
            print("[Exception] {}".format(e))
            print("[sys.exc_info()] {}".format(str(sys.exc_info())))
    
    // [Exception] whynot
    // [sys.exc_info()] (<class 'NameError'>, NameError('whynot',), <traceback object at 0x0000000002573888>)
    

일반적인 경우에는 Exception as e를 쓰자.


[Tistory 원문보기](http://khanrc.tistory.com/53)
