---
layout: post
title: "logstash (2) - with logrotate"
tags: ['Server(Back-end)']
date: 2014-11-07 01:49:00
---
# logstash (2) - with logrotate

[khanrc: logstash with python](http://khanrc.tistory.com/entry/logstash-with-python)에서 파이썬 `logging`모듈로부터 `logstash`를 거쳐 `mongodb`에 로깅하는것까지 성공했다. 허나 `mongodb`의 `insert`속도가 그리 빠르지 않다는 멘토님의 조언에 따라, 몽고디비가 아닌 그냥 `file`에 쓰기로 했다. 

우리 프로젝트는 로그를 기록하면 `data analyzer`가 로그를 읽어서 분석하는 구조로 되어 있다. 데이터 분석기는 일정 주기마다 로그를 읽어서 분석할것이다. 이 때, 이전에 읽은 로그는 놔두고 새로 들어온 로그만 읽어야 한다. 이를 어떻게 처리할 것인가?

## [logrotate](http://linuxcommand.org/man_pages/logrotate8.html)

리눅스 패키지인 `logrotate`라는 것이 있다. 일정 주기별로 로그를 관리해준다. `logrotate`에서는 일정 주기별로 로그를 백업하고, 파일을 rotate하고, 이를 압축하며 메일로 전송하는 등 다양한 관리를 할 수 있다. 이를 이용하여 `logstash`의 로그를 관리해 보자.

먼저 _/etc/logrotate.d/_에서 config파일을 작성하자. _/etc/logrotate.conf_는 default설정 파일이다.
    
    
    $ cd /etc/logrotate.d/
    $ vim newsup
    
    /var/log/newsup/user.log
    {
        rotate 1008
        daily
        missingok
        nocompress
        create
    }
    

자세한 옵션은 따로 구글링하자. 한가지 참고할 점은, `rotate`옵션은 파일의 개수라는 점이다. _logrotate.conf_ 파일에서 `rotate 4` 에 주석으로 4주간 로그를 보관한다고 되어 있는데, 이는 log file rotate 주기가 weekly이기 때문에 4개를 보관하면 자동으로 4주간 보관이 될 뿐이다. 즉, file rotate 주기에 따라 보관 기간이 달라진다. 

그럼, 위 예시에선 daily로 되어 있는데 1008일 동안 보관하는 것인가? 그건 아니다. `logrotate`에서 제공하는 rotate 주기 옵션은 daily, weekly, monthly 이 3가지 뿐이라 10분마다 파일을 로테이트해야 하는 우리 프로젝트와는 맞지 않는다. 해서 강제로 10분으로 설정해 주었다. 즉, 1008 이라는 수치는 (60/10)*24*7 = 1008 일 뿐이다. 즉 1주일 간 보관하는 것이다.

### file rotate 주기 설정

[khanrc: cron, crontab](http://khanrc.tistory.com/entry/cron-crontab%EC%9D%98-%EA%B0%9C%EB%85%90)  
위에서 언급한, `logrotate`에서 제공하는 주기가 아닌 커스텀 설정을 하고 싶으면 어떻게 해야 할까? `cron`에 직접 등록해 주어야 한다. 구글에 검색해 보면 여러 얘기가 나오는데, 처음엔 _/etc/cron.d/_ 에 등록을 하라길래 그렇게 했더니 뭐가 안 되서 결국 `crontab`에 등록했다. 
    
    
    $ sudo -s
    $ crontab -e
    
    ...
    */10  *  *  *  *   root    /usr/sbin/logrotate -f /etc/logrotate.d/newsup
    

이렇게 `crontab -e`로 등록하면 _/var/spool/cron/crontabs/root_ 에 등록이 된다. `crontab -l`로 확인할 수 있다. 위의 cron 명령어는 10분마다 newsup config 파일에 맞게 logrotate를 돌려주는 명령이다.

### problem

문제가 있다. `logstash`에서 파일에 로깅을 할 때 한번 파일을 열고 그 파일에 계속 쓴다. 생각해보면 당연히 그런 형식이겠지. C로 따지면 fopen을 하고 나면 그 파일 포인터로 계속 write를 한다는 소리다. 굳이 close를 해서 로그가 들어올 때마다 파일을 새로 열 필요가 없다. 그러다 보니 문제가 발생한다. `logrotate`에서 로그 파일을 바꿔줘도 새 파일에 로깅하는 것이 아니라 기존 파일에 계속 로깅하는 것이다. 

즉, 예를 들어 logstash가 user.log에 로깅하고 있었다면, logrotate가 이 user.log를 user.log.1로 바꾸고 새로 로깅하라고 user.log를 만들어 준다. 근데 새로 만든 user.log에 로깅하는 것이 아니라 기존의 user.log.1에 로깅하는 것이다. 왜? 파일 포인터는 user.log.1을 가리키고 있으니까. 새로 파일을 열어줘야 하는 것이다.

### solution

이를 해결하기 위해서는 logrotate가 파일을 rotate한 후에 logstash를 재시작해 줘야 한다. 가장 이상적인 구조는, `logstash`의 `Inputs`에 `rabbitmq`를 연결한다. 그래서 `logrotate`가 log file rotate를 하기 전에 `logstash`를 멈추고, file rotate가 끝난 후에 `logstash`를 다시 실행시킨다. 이렇게 하면 `logstash`가 잠시 멈추지만 앞에서 `rabbitmq`가 로그를 받고 있으므로 로그가 사라지지도 않고 모두가 행복하게 된다.

### 산 넘어 산

이제 다 좋은데, `logstash`가 `service`가 아니라는 문제가 남았다. `logrotate`에서 file rotate앞뒤로 logstash를 재시작해 주려면 service logstash stop, start, restart 따위의 명령어를 써야 할 텐데 그게 없는 것이다. 내가 직접 하는거면 ps aux | grep logstash 해서 kill pid로 죽이면 그만이지만 그게 아니니 문제가 된다. 

그래서 처음엔 logstash를 service에 등록하려고 삽질을 했다. <https://github.com/bloonix/logstash-pkgs> 요런 걸 보면 그런 짓을 해주는 걸 볼 수 있다. 나도 처음엔 그쪽으로 생각했는데 실패해서 방법을 바꾸기로 했다. 저 소스도 그렇고 다른 소스도 그렇고 공홈에서 하란대로 설치하는거랑 안 맞는 듯. 추후 여유가 있으면 apt-get으로 설치해보고 테스트 해봐야겠다.

### python: os.popen

서비스를 등록하는 삽질을 해보는 것도 충분히 의미있는 삽질(?) 이겠지만, 지금은 시간이 없어 멘토님의 조언대로 파이썬을 활용하기로 했다. 파이썬에는 `os.popen`이라는 명령어가 있는데, 커맨드라인에 명령어를 치듯이 그대로 할 수 있다. logstash-controller.py를 만들었다.
    
    
    import os
    import sys
    
    def start():
        os.popen("sudo /home/khanrc/tworoom/logstash/bin/logstash -f /home/khanrc/tworoom/logstash/file.conf &", "r")
    
    def stop():
        p = os.popen("ps aux | grep bin/logstash", "r")
        s = p.read()
        d = s.split("\n")
        pid = int(d[0].split()[1])
        os.popen("kill " + `pid`)
    
    ...
    
    if sys.argv[1] == "start":
        start()
    elif sys.argv[1] == "stop":
        stop()
    elif sys.argv[1] == "restart":
        stop()
        start()
    

대충 이렇게 되어 있다. 훌륭하게 작동한다!  
이제 이걸 활용해서, 위에서 언급했었던 file rotate 앞뒤로 stop / start를 해 주어야 한다. logrotate config 파일에 아래 스크립트를 추가해주자.
    
    
    postrotate
        python /home/khanrc/tworoom/logstash/logstash-controller.py start
    endscript
    prerotate
        python /home/khanrc/tworoom/logstash/logstash-controller.py stop
    endscript
    

자, 이제 잘 수 있다!


[Tistory 원문보기](http://khanrc.tistory.com/68)
