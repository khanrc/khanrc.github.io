---
layout: post
title: "REST API 서버 제작기 (2) - Flask 서버 올리기"
tags: ['Web']
date: 2014-09-24 23:46:00
---
# REST API 서버 제작기 (2) - Flask 서버 올리기

API 서버는 플라스크로 제작하기로 했다.

[Ubuntu에 개발환경 셋팅하기 (1)](http://khanrc.tistory.com/entry/Ubuntu%EC%97%90-%EA%B0%9C%EB%B0%9C%ED%99%98%EA%B2%BD-%EC%85%8B%ED%8C%85%ED%95%98%EA%B8%B0-1)  
[Ubuntu에 개발환경 셋팅하기 (2)](http://khanrc.tistory.com/entry/Ubuntu%EC%97%90-%EA%B0%9C%EB%B0%9C%ED%99%98%EA%B2%BD-%EC%85%8B%ED%8C%85%ED%95%98%EA%B8%B0-2)  
이 서버에 Flask를 올리고 API 서버를 구현한다.

## Flask quick start

<http://flask-docs-kr.readthedocs.org/>

### Installation

일단 설치부터.

일반적으로 virtualenv 사용을 권장하지만, 본 프로젝트에서는 사용하지 않는다.
    
    
    $ sudo pip install Flask
    [sudo] password for khanrc:
    Requirement already satisfied (use --upgrade to upgrade): Flask in /usr/local/lib/python2.7/dist-packages
    Requirement already satisfied (use --upgrade to upgrade): Werkzeug>=0.7 in /usr/local/lib/python2.7/dist-packages (from Flask)
    Requirement already satisfied (use --upgrade to upgrade): Jinja2>=2.4 in /usr/local/lib/python2.7/dist-packages (from Flask)
    Requirement already satisfied (use --upgrade to upgrade): itsdangerous>=0.21 in /usr/local/lib/python2.7/dist-packages (from Flask)
    Requirement already satisfied (use --upgrade to upgrade): markupsafe in /usr/local/lib/python2.7/dist-packages (from Jinja2>=2.4->Flask)
    Cleaning up...
    

으잉.. 이미 깔려 있다.

### Quick start

역시 시작은 Hello world지!  
위 도큐먼트에 나와있는대로 그대로 따라하면 된다.
    
    
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def hello_world():
        return 'Hello World!'
    
    if __name__ == '__main__':
        app.run()
    

이렇게 하면 localhost에 올라간다. 나는 지금 서버에 곧바로 작업하고 있으므로 접근할 수 있게 호스트를 바꿔주자. 
    
    
    app.run(host='0.0.0.0', debug='True')
    

debug='True' 로 해 두면 코드 변경을 자동으로 감지하여 알아서 리로드하고, 코드 에러가 나면 문제 해결을 위한 디버거를 제공한다. 실제로 돌릴 때는 꺼야 한다(고 한다).

이렇게 하면 Hello World! 를 볼 수 있다!  
5000번 포트를 열어주는 것도 잊지 말자.
    
    
    $ python app.py
     * Running on http://0.0.0.0:5000/
     * Restarting with reloader
    

이렇게 실행한다.

### Run, always run!

근데 저렇게 실행하면 다른 작업을 하기 위해서는 프로세스를 죽여야 한다.  
그럴 필요 없도록 백그라운드에서 돌리려면 `&` 를 붙여주면 된다.
    
    
    $ python app.py &
    [1] 4981
    

pid 4981로 실행된다고 알려준다.  
그럼 ps ux를 쳐보면
    
    
    $ ps ux
    USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
    khanrc    4857  0.0  0.0 102132  1852 ?        S    22:14   0:00 sshd: khanrc@pts/1
    khanrc    4858  0.7  0.0  27264  8548 pts/1    Ss   22:14   0:00 -bash
    khanrc    4981  1.0  0.0  58348 15536 pts/1    S    22:14   0:00 python app.py
    khanrc    4986  1.3  0.0 138424 15988 pts/1    Sl   22:14   0:00 /usr/bin/python app.py
    khanrc    5071  0.0  0.0  18160  1272 pts/1    R+   22:15   0:00 ps ux
    

백그라운드에서 돌아가고 있다.

실제로 http request를 날려도 정상적으로 반응한다.
    
    
    $ curl http://ip-address:5000/
    Hello World!!
    

그런데 이렇게 `&`로 하더라도 로그아웃을 하면 또 프로세스가 죽는다1.  
이를 위해 사용하는것이 바로 `nohup`이다.  
[nohup 사용법](http://jmnote.com/wiki/Nohup_%EC%82%AC%EC%9A%A9%EB%B2%95)

> nohup이란?  
no hang up. 프로세스 중단(hangup)을 무시하고 명령어를 실행하는 명령어  
표준출력을 nohup.out(또는 다른 곳)으로 돌림
    
    
    $ nohup python app.py &
    

이렇게 실행을 해 두면 standard out을 nohup.out 따위의 파일로 출력할 수 있어 편리하다.  
다른 파일로 출력하고 싶다면 
    
    
    $ nohup python app.py > app.log &
    

이런식으로 하면 된다.

> 그런데 이렇게 해도 이 프로세스가 죽을 수 있지 않은가?  
이를 대비하기 위해 쉘 스크립트로 while(1)을 활용해 app.py를 계속 실행시키자. app.py가 죽으면 자동으로 while이 돌아 다시 실행시켜준다!  
이 쉘 스크립트를 백그라운드에 돌려 놓으면, 완벽하다.

* * *

  1. 내가 테스트한 우분투 12.04에서는 죽지 않았는데, 이게 언젠가부터 안죽게 변경되었다고 한다.↩


[Tistory 원문보기](http://khanrc.tistory.com/42)
