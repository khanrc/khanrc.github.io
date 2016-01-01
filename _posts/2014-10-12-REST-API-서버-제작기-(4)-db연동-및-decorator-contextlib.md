---
layout: post
title: "REST API 서버 제작기 (4) - db연동 및 decorator, contextlib"
tags: ['Server(Back-end)']
date: 2014-10-12 20:46:00
---
# REST API 서버 제작기 (4) - decorator 및 db연동

최근, 블로그에 글을 컴팩트하게 쓰기로 마음먹으면서 이 시리즈(?)를 왜 쓰는가에 대한 회의가 좀 있지만 일단 시작한 것이니 만큼 끝까지 쓰기로 한다. 시작했으면 끝을 봐야지. 내용이 좀 짧아지는건 어쩔 수 없다.

## decorator

[khanrc: decorator와 closure](http://khanrc.tistory.com/entry/decorator%EC%99%80-closure)  
[khanrc: decorator (2) - extension](http://khanrc.tistory.com/entry/decorator-2-extension)

`before_filter`로 만들어 두었던 것을 `decorator`로 변경했다. 유저 토큰 검사는 데코레이터와 굉장히 잘 어울린다.
    
    
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
    

## mongodb

[khanrc: mongoDB](http://khanrc.tistory.com/category)  
그간 써보고 싶었던 `mongodb`를 써 보았다. 위 링크를 보면 알겠지만 `mongokit`이라고 ORM도 있는데, ORM을 그다지 선호하지 않아 `pymongo`를 사용하였다. `flask`의 extension인 `flask-pymongo`가 존재하는데 별로 필요 없어 보인다.
    
    
    from flask.ext.pymongo import PyMongo
    import pymongo
    
    app = Flask(__name__)
    app.config['MONGO_DBNAME'] = 'newsup'
    # app.config['MONGO_HOST'] = 'localhost' # expect localhost by default.
    mongo = PyMongo(app)
    
    def getSeq(name):
        ret = mongo.db.ikeys.find_and_modify(
                query={ '_id' : name },
                update={ '$inc': { 'seq': 1 } },
                new=True
            )
    
        return ret.get('seq')
    
    @app.route('/users', methods=['POST'])
    def add_user():
        user_token = request.headers.get('user_token')
        if user_token is None:
            return Response(response="user_token is None",
                            status=401)
    
        try:
            ret = mongo.db.users.insert({
                '_id': user_token,
                'user_id': getSeq('user_id')
            })
    
            if ret == False:
                return "fail!!!!"
        except pymongo.errors.DuplicateKeyError, e:
            return Response(response="{error_code:1, error_msg:'중복된 유저가 있음'}",
                            status=200)
        except:
            print "치명적인 에러 in add_user, insert user_token: " + str(sys.exc_info())
            return Response(response="{error_code:-1}", mimetype='application/json', status=200)
    
    
        return Response(response="{error_code:0}", mimetype='application/json', status=201)
    

`getSeq()`는 auto_increment를 구현한 것. ikeys 컬렉션에 seq를 저장해 놓고 매 호출마다 증가시킨다.

다만 위처럼 하면 `json` 인식이 제대로 안 되는데, `response`를 저렇게 쓰면 안 된다.
    
    
    return Response(response="""{"error_code":-1}""", mimetype='application/json', status=200)
    

이렇게 `string`은 `""`로 감싸줘야 한다.

## mysql (mariadb)

python-mysql 모듈도 굉장히 다양하게 존재한다. `mysql`에서 공식적으로 지원하는 `mysql.connector`, 가장 많이 쓰는 것 같은 `mysqldb` 등등. 이 모듈들의 성능 비교를 한 포스트가 있다.  
[Comparing python/mysql drivers](http://mypysql.blogspot.kr/2013/05/comparaison-of-pythonmysql-connectors.html)

성능이 좋다고 하고, 많이들 사용하는 `mysqldb`를 사용해 보기로 했다. `mysqldb`는 mysql의 C API를 직접적으로 사용하는 `_mysql`모듈의 래퍼다.

### install

[참고: Is MySQLdb hard to install?](http://mysql-python.blogspot.kr/)
    
    
    sudo apt-get install build-essential python-dev libmysqlclient-dev
    sudo pip install MySQL-python
    

단, 우리는 `mariadb`이기 때문에 `libmysqlclient-dev`가 아니라 `libmariadbclient-dev`를 설치해야 한다.  
<http://stackoverflow.com/questions/22949654/mysql-config-not-found-when-installing-mysqldb-python-interface-for-mariadb-10-u> 참고.
    
    
    sudo apt-get install libmariadbclient-dev
    sudo pip install MySQL-python
    

이러면 설치가 된다.

### usage
    
    
    import MySQLdb as mdb
    import _mysql_exceptions
    import sys
    
    con = mdb.connect(host='host', user='user', passwd='passwd', db='db')
    cur = con.cursor()
    
    try:
        p = 'hello11'
        s = """INSERT INTO USER (user_token) VALUES ('%s')""" % (p)
        print s
        cur.execute(s)
        con.commit()
    except _mysql_exceptions.IntegrityError, e:
        print "IntegrityError : ", e
        con.rollback()
    except:
        print "error!! rollback!!" + str(sys.exc_info())
        con.rollback()
    
    cur.execute("""SELECT * FROM USER""")
    print cur.fetchall()
    

처음에 이렇게 코딩을 했더니, 처음엔 잘 되다가 시간이 지나니 `OperationalError: (2006, 'MySQL server has gone away')` 이런 에러가 났다. 구글링을 해 보니 connection이 끊겨서 나는 에러라고 하더라. 생각해 보니 저렇게 한번 커넥트를 해주면 되는게 아니라, 매 요청시마다 커넥트를 해 주고 끝나면 닫아줘야 한다. 커넥션이 타임아웃되어 에러가 났던 것.

그래서 매 요청마다 커넥트와 클로즈를 해주자니 너무 코드가 지저분해진다. 그래서 `contextlib`을 쓰기로 했다.

### contextlib

[khanrc: contextlib](http://khanrc.tistory.com/entry/contextlib) 참고.

위 링크에도 나와있지만
    
    
    from contextlib import contextmanager
    
    @contextmanager
    def db_connect():
        con = mdb.connect(host, user, passwd, db)
        cur = con.cursor()
    
        try:
            yield (con, cur)
        finally:
            cur.close()
            con.close()
    
    with db_connect() as (con, cur):
        p = """ '%s' """ % user_token
        s = """INSERT INTO USER (user_token) VALUES (%s)""" % (p)
        cur.execute(s)
        con.commit()
    

요렇게 쓴다.


[Tistory 원문보기](http://khanrc.tistory.com/56)
