---
layout: post
title: "Nginx deploy! (+uWSGI +flask)"
tags: ['Web']
date: 2014-11-15 19:32:00
---
# Nginx deploy! (+uWSGI +flask)

## uWSGI

[uWSGI](http://flask.pocoo.org/docs/0.10/deploying/uwsgi/)  
[Flask(uWSGI)를 nginx에 연결하기](http://haruair.com/blog/1900)

uWSGI를 먼저 설치하고 실행한다.
    
    
    $ pip install uwsgi
    

나는 ini파일을 만들어 실행했다. python path 문제 때문에 그래야만 했다. virtualenv를 안 써서 그런지 자꾸 python path를 못 찾아서, sys.path를 치면 나오는 path 들을 죄다 갖다 박아줬다. 다른 실행 옵션들은 위 링크를 참조하자.
    
    
    [uwsgi]
    chdir=/home/khanrc/tworoom/flask
    chmod-socket=666
    callable=app
    module=app
    socket=/tmp/uwsgi.sock
    pythonpath=/usr/lib/python2.7
    pythonpath=/usr/lib/python2.7/plat-linux2
    pythonpath=/usr/lib/python2.7/lib-tk
    pythonpath=/usr/lib/python2.7/lib-old
    pythonpath=/usr/lib/python2.7/lib-dynload
    pythonpath=/usr/local/lib/python2.7/dist-packages
    pythonpath=/usr/lib/python2.7/dist-packages
    pythonpath=/usr/lib/python2.7/dist-packages/PIL
    pythonpath=/usr/lib/python2.7/dist-packages/gtk-2.0
    pythonpath=/usr/lib/pymodules/python2.7
    

간단하게 적었지만 이 과정에서 엄청 고생했다. 이제, 실행해보자.
    
    
    $ uwsgi uwsgi.ini &
    

당연한 얘기지만 &amp;를 빼면 foreground로 실행된다.

## Nginx

설치부터 하자.
    
    
    $ apt-get install nginx-full
    

설치하고 나면 설정파일을 수정해서 uwsgi와 연결해 주자.
    
    
    server {
        listen 80;
        server_name ip_address;
    
        location / {
            try_files $uri @app;
        }
    
        location @app {
            include uwsgi_params;
            uwsgi_pass unix:/tmp/uwsgi.sock;
        }
    }
    

이제 nginx와 uwsgi, flask가 전부 연결되었다.
    
    
    $ /etc/init.d/nginx restart
    

Done!

## Problem

어? 다 된 줄 알았는데 request에 header가 없다. -_-;;  
뭔가 하고 한참 삽질했는데, 보니까 우리 헤더 이름이 `user_token`인데 이게 규약에 어긋나는 것 같다. 자세한 건 여기를 참조하자: [List of HTTP header fields](http://en.wikipedia.org/wiki/List_of_HTTP_header_fields)

flask를 그냥 `python app.py`로 실행하면 자체 서버로 실행이 되는데, 이러면 이 자체 서버가 헤더를 바꿔주는 것 같다. user_token으로 보내도 `User-Token`으로 들어온다. 그런데 nginx는 그렇지 않다. user_token으로 보내면 그냥 버려버린다 -_-;;

그래서 user_token을 User-Token으로 변경하는 것으로 마무리했다.


[Tistory 원문보기](http://khanrc.tistory.com/73)
