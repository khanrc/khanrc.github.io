---
layout: post
title: "decorator (2) - extension"
tags: ['Python']
date: 2014-10-06 10:28:00
---
# decorator (2) - extension

확장이라고 하기에는 별 거 없지만, 실제로 적용하면서 알게된 몇가지 더 정리해 보았다.  
`@wraps`는 항상 적용해 주는 게 좋은 것 같고, 그 이외에는 적용하면서 알아가면 되는 부분이지 미리 학습할 필요까진 없어 보인다.

## [@wraps](https://gist.github.com/shoveller/b4d2e1e6d33906f2a667)

플라스크에 데코레이터를 적용하려고 찾아봤더니, `@wraps`라는 게 보인다. 이게 뭐지?
    
    
    from functools import wraps
    
    def without_wraps(func):
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return __wrapper
    
    def with_wraps(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return __wrapper
    
    
    @without_wraps
    def my_func_a():
        """Here is my_func_a doc string text."""
        pass
    
    @with_wraps
    def my_func_b():
        """Here is my_func_b doc string text."""
        pass
    
    '''
    # Below are the results without using @wraps decorator
    print my_func_a.__doc__
    >>> None
    print my_func_a.__name__
    >>> __wrapper
    
    # Below are the results with using @wraps decorator
    print my_func_b.__doc__
    >>> Here is my_func_b doc string text.
    print my_func_b.__name__
    >>> my_func_b
    '''
    

위 코드를 찬찬히 읽어 보면 뭔지 알 수 있다. 기존 함수를 데코레이터로 래핑하게 되면, 데코레이트된 함수의 속성을 요청하면 기존 함수의 속성이 나오는 게 아니라 데코레이트된 래퍼의 속성이 나오는 것이다. `@wraps`를 사용하면 이 문제를 해결할 수 있다.

## pass parameter on func

래퍼인 데코레이터에서 감싸진 원 함수로 파라메터를 넘기고 싶다면?  
`kwargs`로 넘기면 된다.
    
    
    from inspect import getargspec
    
    # decorator.
    def usertoken_required(func):
        @wraps(func)
        def decorated(*args, **kwargs):
            user_token = request.headers.get('user_token')
            cur.execute("""SELECT * FROM USER WHERE user_token='%s'""" % user_token)
            user = cur.fetchone()
    
            if user is None:
                return Response(response="user_token is wrong!", 
                                status=401)
    
            argspec = getargspec(func)
            if argspec[2] is not None: # kwargs를 받는 함수에게만 전달
                kwargs['user_info'] = user
    
            return func(*args, **kwargs)
    
        return decorated
    
    
    @app.route('/')
    @usertoken_required
    def hello_world(**kwargs):
        return 'Hello,' + request.headers.get('user_token') + ' user:' + str(kwargs['user_info'])
    

usertoken을 검사하는 데코레이터를 만들었는데, 데코레이터에서 찾은 유저 정보를 버리기가 아까워 `kwargs`에 담아 넘겼다. 감싸지는 원 함수인 `func`에서 `kwargs`를 받지 않는 경우를 대비해서, `getargspec`으로 `args`의 스펙을 받아 `kwargs`를 받는 경우에만 user_info를 넘기도록 했다.

## [more in flask: return](http://flask.pocoo.org/docs/0.10/patterns/viewdecorators/)

플라스크에서 데코레이터를 찾아보면 적절한 예제 페이지가 하나 나온다.
    
    
    from functools import wraps
    from flask import g, request, redirect, url_for
    
    def login_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if g.user is None:
                return redirect(url_for('login', next=request.url))
            return f(*args, **kwargs)
        return decorated_function
    
    
    @app.route('/secret_page')
    @login_required
    def secret_page():
        pass
    

이 외에도 링크에 들어가 보면 실제로 데코레이터를 어떻게 사용하는지 적절한 예제들이 잘 나와 있다. 데코레이터를 쓸 생각이면 한번 살펴보도록 하자. 

위 예제에서는 `return` 이 있는 함수를 데코레이트 하는 방법을 볼 수 있다.

## decorator with parameter

마찬가지로 위 플라스크 페이지에서 가져온 예제다. `flask`의 `@app.route`처럼, 데코레이터에 파라메터를 넘겨서 처리하고 싶다면?
    
    
    from functools import wraps
    from flask import request
    
    def templated(template=None):
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                template_name = template
                if template_name is None:
                    template_name = request.endpoint \
                        .replace('.', '/') + '.html'
                ctx = f(*args, **kwargs)
                if ctx is None:
                    ctx = {}
                elif not isinstance(ctx, dict):
                    return ctx
                return render_template(template_name, **ctx)
            return decorated_function
        return decorator
    
    @app.route('/')
    def index():
        return render_template('index.html', value=42)
    
    @app.route('/')
    @templated('index.html')
    def index():
        return dict(value=42)
    
    @app.route('/')
    @templated()
    def index():
        return dict(value=42)
    

이렇게 한번 더 감싸줌으로써 처리할 수 있다.

## 참고

[왜 파이썬 데코레이터를 만들때, @wraps어노테이션을 쓰는 것을 권장하는 걸까?](https://gist.github.com/shoveller/b4d2e1e6d33906f2a667)  
[Flask: View Decorators](http://flask.pocoo.org/docs/0.10/patterns/viewdecorators/)


[Tistory 원문보기](http://khanrc.tistory.com/49)
