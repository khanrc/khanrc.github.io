---
layout: post
title: "contextlib"
tags: ['Python']
date: 2014-10-12 20:26:00
---
# contextlib

데코레이터와 마찬가지로 파이썬에서 지원하는 강력한 기능중 하나. 파이썬을 쓰다보면 `with`라는 키워드를 볼 수 있는데, 이에 관련된 라이브러리가 바로 `contextlib`이다. `with`는 시작과 끝이 있는 경우에 사용하는데, file이나 database처럼 open 또는 connect 후 close가 필요한 경우가 대표적이다.

## intro
    
    
    import time
    
    class demo:
        def __init__(self, label):
            self.label = label
    
        def __enter__(self):
            self.start = time.time()
    
        def __exit__(self, exc_ty, exc_val, exc_tb):
            end = time.time()
            print('{}: {}'.format(self.label, end - self.start))
    
    with demo('counting'):
        n = 10000000
        while n > 0:
            n -= 1
    
    # counting: 1.36000013351
    

보면 알겠지만 `__enter__`는 `with`문이 시작할 때, `__exit__`는 끝날 때 실행된다.

이제 이걸 `contextlib`을 사용해서 간단하게 바꿀 수 있다.
    
    
    from contextlib import contextmanager
    import time
    
    @contextmanager
    def demo(label):
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            print('{}: {}'.format(label, end - start))
    
    with demo('counting'):
        n = 10000000
        while n > 0:
            n -= 1
    
    # counting: 1.32399988174
    

`yield`가 `with`문이 감싸는 코드를 실행시킨다. `yield`를 통해서 오브젝트를 전달할 수 있다.

## closing

앞에서 말한 것처럼, `with`문은 close와 함께 많이 활용되고 이를 위해 `closing`이란 게 있다.
    
    
    from contextlib import closing
    import MySQLdb
    
    con = MySQLdb.connect("host", "user", "pass", "database")
    with closing(con.cursor()) as cur:
        cur.execute("somestuff")
        results = cur.fetchall()
    
        cur.execute("insert operation")
        con.commit()
    
    con.close()
    

db connect할 때 이렇게 많이 쓰이는 것 같다.

## apply
    
    
    from contextlib import contextmanager
    
    @contextmanager
    def mysql_connect():
        con = mdb.connect(host, user, passwd, db)
        cur = con.cursor()
    
        try:
            yield (con, cur)
        finally:
            cur.close()
            con.close()
    
    with mysql_connect() as (con, cur):
            cur.execute("""SELECT * FROM USER""")
            tu = cur.fetchall()
            con.commit()
    

이렇게 적용하는 것이 가장 심플해 보인다. `yield`의 용법과 2개 이상의 오브젝트를 전달하는 방법도 참고하자.

## 참고

[Python - 수준 있는 디자인 패턴 (Advanced Design Patterns in Python)](http://jhproject.tistory.com/103)

[python-docs: 27.7. contextlib -- Utilities for with-statement contexts](https://docs.python.org/2/library/contextlib.html) : 공식 문서  
[MySQLdb &amp; closing](http://stackoverflow.com/questions/5669878/python-mysqldb-when-to-close-cursors)


[Tistory 원문보기](http://khanrc.tistory.com/55)
