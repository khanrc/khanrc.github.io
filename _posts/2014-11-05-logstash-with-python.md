---
layout: post
title: "logstash with python"
tags: ['Server(Back-end)']
date: 2014-11-05 11:34:00
---
# logstash

데이터 분석을 하려면 데이터가 있어야 한다. 프로젝트마다 다르겠지만, 개인화 프로젝트에서는 그 데이터를 당연히 제공하는 서비스에서 수집한다. 소마 프로젝트 또한 마찬가지로 데이터 수집 과정이 있는데, 데이터 수집은 자연스럽게 로그 수집으로 이어진다. 그러면서 접하게 된 것이 바로 `logstash`이다.

## log aggregator

여러 노드(인스턴스)로부터 로그 데이터를 모아주는 프레임워크를 `log aggregator`라 부른다. 클라우드 환경이 대두되고, 빅데이터가 떠오르면서 자연스럽게 필요하게 된 모듈이라고 할 수 있다.

`log aggregator`도 종류가 다양한데, facebook에서 사용해서 유명해진 `scribe`, cloudera에서 제작하여 현재 apache의 top level project인 `flume`, ruby와 c로 짜여진 `fluentd`, 사용이 간편하다고 하는 `logstash`등이 있다. 본 프로젝트에서 처음엔 `flume`을 고려했으나, 사용의 편의성을 위해 `logstash`를 사용하기로 했다.

## [getting start](http://logstash.net/docs/1.4.2/tutorials/getting-started-with-logstash)

공식 문서를 참조하자:

### intro

`Logstash`는 로그를 받고, 처리하고, 출력한다. `Elasticsearch`를 백엔드 데이터 스토리지로 사용하고, `kibana`를 프론트엔드 리포팅 툴로 활용하면서, `logstash`는 그 동력으로서 로그를 수집하고 분석한다. 간단한 조작을 통해 강력한 기능들을 활용할 수 있다. 당장 시작하자!

### prerequisite: java

`logstash`는 자바로 짜여졌다. 그래서 돌리려면 자바가 필요하다. `java -version`을 통해 확인해보자.
    
    
    $ java -version
    java version "1.7.0_65"
    OpenJDK Runtime Environment (IcedTea 2.5.1) (7u65-2.5.1-4ubuntu1~0.12.04.2)
    OpenJDK 64-Bit Server VM (build 24.65-b04, mixed mode)
    

### Up and Running!

#### Logstash in two commands

일단 `logstash`를 받자.
    
    
    $ curl -O https://download.elasticsearch.org/logstash/logstash/logstash-1.4.2.tar.gz
    

만약 `curl`이 없다면 깔아라.
    
    
    $ sudo apt-get install curl
    

그리고 나서 압축을 풀고 실행해보자.
    
    
    $ tar zxvf logstash-1.4.2.tar.gz
    $ cd logstash-1.4.2
    
    $ bin/logstash -e 'input { stdin { } } output { stdout {} }'
    

그리고 아무거나 쳐 보면, 그대로 로깅이 된다.
    
    
    hello world
    2014-10-31T15:02:10.201+0000 c826a788-0110-4c31-8ac0-db1dab0fda32 hello world
    

위 명령어를 되새겨보자. `stdin`으로 입력받고, `stdout`으로 출력했다. `-e`커맨드는 config를 `cli`에서 곧바로 입력할 수 있게 해준다.

이제, 좀 더 fancy한 걸 해보자.
    
    
    $ bin/logstash -e 'input { stdin { } } output { stdout { codec => rubydebug } }'
    
    hi
    {
           "message" => "hi",
          "@version" => "1",
        "@timestamp" => "2014-10-31T15:12:51.619Z",
              "host" => "c826a788-0110-4c31-8ac0-db1dab0fda32"
    }
    

보다시피, `codec`을 추가한 것만으로 출력 포맷을 바꿨다. 이렇게 간단한 설정을 통해 입력과 출력 및 필터링을 할 수 있다.

### Moving On

`logstash`의 config를 네가지로 분류하자면 `Inputs`, `Outputs`, `Codecs` 그리고 `Filters`다. 아래에 대표적인 케이스를 소개한다:

  * Inputs: 어디서 로그를 입력받을 것인가?
    * file
    * syslog
    * redis
    * lumberjack
  * Filters: `Inputs`와 `Outputs`사이에서의 중간 프로세싱 과정
    * grok: 임의의 log 텍스트를 구조화한다
    * mutate: rename, replace, remove, modify fields …
    * drop: 어떤 이벤트를 통째로 버린다
    * clone: 카피한다 - 필드에 수정을 가해서 카피할수도 있다.
    * geoip: GEOgraphical location of IP. 위치정보를 추가한다. `kibana`에서 display할 수 있다.
  * Outputs: 어디에 출력할 것인가?
    * elasticsearch
    * file
    * [graphite](http://graphite.wikidot.com/): 데이터를 저장하고 시각적으로 보여주는 오픈소스 툴
    * statsd
  * Codecs: 어떻게 출력할 것인가?
    * json
    * multiline

### More fun

#### Persistent Configuration file

config를 `-e`로 줄 수 있지만, 당연히 file로 설정할 수도 있다. `logstash-simple.conf`라는 파일을 만들고 로그스태시 디렉토리에 저장하자.
    
    
    input { stdin { } }
    output {
      stdout { codec => rubydebug }
    }
    

그리고 로그스태시를 실행하자:
    
    
    $ bin/logstash -f logstash-simple.conf
    

`-e`로 설정을 준 것과 동일하게 잘 작동하는 것을 볼 수 있다. 즉, `-e`는 커맨드라인에서 설정을 읽고 `-f`는 파일에서 설정을 읽는다.

#### Filters

필터를 적용해 보자. `grok` 필터는, 위에서도 언급했지만, 대표적인 로그들을 자동으로 구조화해준다.
    
    
    input { stdin { } }
    
    filter {
      grok {
        match => { "message" => "%{COMBINEDAPACHELOG}" }
      }
      date {
        match => [ "timestamp" , "dd/MMM/yyyy:HH:mm:ss Z" ]
      }
    }
    
    output {
      stdout { codec => rubydebug }
    }
    

이렇게 config파일을 수정하고 로그스태시를 실행해서 아래 로그를 입력하자:
    
    
    $ bin/logstash -f logstash-filter.conf
    
    127.0.0.1 - - [11/Dec/2013:00:01:45 -0800] "GET /xampp/status.php HTTP/1.1" 200 3891 "http://cadenza/xampp/navi.php" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:25.0) Gecko/20100101 Firefox/25.0"
    

그럼 이런 결과를 얻을 수 있다:
    
    
    {
            "message" => "127.0.0.1 - - [11/Dec/2013:00:01:45 -0800] \"GET /xampp/status.php HTTP/1.1\" 200 3891 \"http://cadenza/xampp/navi.php\" \"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:25.0) Gecko/20100101 Firefox/25.0\"",
         "@timestamp" => "2013-12-11T08:01:45.000Z",
           "@version" => "1",
               "host" => "cadenza",
           "clientip" => "127.0.0.1",
              "ident" => "-",
               "auth" => "-",
          "timestamp" => "11/Dec/2013:00:01:45 -0800",
               "verb" => "GET",
            "request" => "/xampp/status.php",
        "httpversion" => "1.1",
           "response" => "200",
              "bytes" => "3891",
           "referrer" => "\"http://cadenza/xampp/navi.php\"",
              "agent" => "\"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:25.0) Gecko/20100101 Firefox/25.0\""
    }
    

위 예시에서 `grok`필터 뿐만 아니라 `date`필터도 사용되었다. 위에서 볼 수 있다시피, 로그의 timestamp를 캐치한다.

### with Elasticsearch?

`logstash`는 `Elasticsearch`와 강력하게 연동되지만, 공식 문서에서도 관련하여 여러가지를 소개하지만 우리 프로젝트에서 엘라스틱서치를 사용하지 않기 때문에 그 내용은 따로 다루지 않았다. 필요하면 찾아보도록 하자.

## Apply

<http://logstash.net/docs/1.4.2/>  
위 도큐먼트에서 어떤식으로 각 `plugin`을 연동시켜야 할지 찾아볼 수 있다.

### [milestone](http://logstash.net/docs/1.4.2/plugin-milestones)

각 플러그인을 보면 `milestone level`이라는 것이 있다. 위 링크를 참조하자. 높을수록 좋은 것이고 2 이상이면 안정적인 것 같다.

### Outputs: mongodb

<http://logstash.net/docs/1.4.2/outputs/mongodb>  
출력을 어떻게 해야 할 지는 조금 더 고민해 봐야 할 문제지만 일단 `mongodb`에 출력하기로 한다. 도큐먼트를 참조하자. 먼저 `contrib plugin`을 설치해야 한다.
    
    
    $ bin/plugin install contrib
    

그리고 config파일을 작성하자
    
    
    $ vim logstash-mongo.conf
    
    input { stdin { } }
    
    output {
        mongodb {
            collection => "logstash"
            database => "test"
            uri => "mongodb://localhost"
        }
    }
    

`-t`로 config파일이 제대로 작성되었는지 확인할 수 있다.
    
    
    $ bin/logstash -t -f logstash-mongo.conf
    Using milestone 2 output plugin 'mongodb'. This plugin should be stable, but if you see strange behavior, please let us know! For more information on plugin milestones, see http://logstash.net/docs/1.4.2/plugin-milestones {:level=>:warn}
    Configuration OK
    

이제 `-t`를 빼고 실행시켜보면 `mongodb`와 정상적으로 연동된다.

### Inputs: python-logstash

`python`의 `logging`모듈과 `logstash`를 연결해 주는 [python-logstash](https://pypi.python.org/pypi/python-logstash/0.4.2)라는 라이브러리가 있다.
    
    
    $ sudo pip install python-logstash
    

먼저 테스트를 해 보자. example에서 udp, 5959번 포트로 로그를 보내므로 그에 맞게 config파일을 설정해주자.
    
    
    input {
        udp {
            port => 5959
        }
    }
    
    output {
      stdout { codec => rubydebug }
    }
    

example을 좀 들여다 보면,
    
    
    import logging
    import logstash
    
    test_logger = logging.getLogger('python-logstash-logger')
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(logstash.LogstashHandler(host, 5959, version=1))
    
    test_logger.info('python-logstash: test logstash info message.')
    
    ...
    

`logging`모듈을 동일하게 사용하되 handler만 logstashHandler로 설정해 주면 된다는 것을 알 수 있다. python-logstash는 `Inputs`로 `udp`와 `tcp` 두 가지를 지원한다.

그럼 이제 로그스태시를 실행시키고 example.py를 돌리면
    
    
    $ bin/logstash -f logstash-simple.conf
    Using milestone 2 input plugin 'udp'. This plugin should be stable, but if you see strange behavior, please let us know! For more information on plugin milestones, see http://logstash.net/docs/1.4.2/plugin-milestones {:level=>:warn}
    {
           "message" => "{\"host\": \"c826a788-0110-4c31-8ac0-db1dab0fda32\", \"logger\": \"python-logstash-logger\", \"type\": \"logstash\", \"tags\": [], \"path\": \"test.py\", \"@timestamp\": \"2014-11-04T08:44:50.616251Z\", \"@version\": 1, \"message\": \"python-logstash: test logstash error message.\", \"levelname\": \"ERROR\"}",
          "@version" => "1",
        "@timestamp" => "2014-11-04T08:44:50.618Z",
              "host" => "127.0.0.1"
    }
    
    ...
    

이렇게 로그를 잘 받아온다.

### in flask

실제로 `flask`에 적용해보자.
    
    
    input {
        udp {
            port => 5959
        }
    }
    
    output {
        mongodb {
            collection => "logstash"
            database => "test"
            uri => "mongodb://localhost"
        }
    }
    

`Inputs`는 `udp`로, `Outputs`는 `mongodb`로 설정했다.  
그리고 example처럼 코드를 적어주자.
    
    
    import logstash
    import logging
    
    ...
    
    # 로거는 이 프로세스가 죽을때까지 누적된다.
    # 따라서 아래와 같이 처음 로거를 할당한건지를 체크해서 한 번만 설정해 줘야 한다.
    # 그렇지 않으면, level은 상관 없지만 handler는 계속 누적되어 핸들러가 계속 늘어나서 한번만 로깅해도 여러개가 찍히게 된다.
    logger = logging.getLogger('logstash-logger')
    if len(logger.handlers) == 0: 
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        lh = logstash.LogstashHandler('localhost', 5959, version=1)
        lh.setFormatter(formatter)
        logger.addHandler(lh)
    
    logger.info(json.dumps(js))
    

example에서는 한번 logger를 할당해서 사용한 후 프로세스가 종료되므로 상관 없지만, `flask`에서는 프로세스가 계속 살아있다. 이 때문에, 주석에서 설명한 것처럼 handler가 여러개 세팅되지 않도록 조심해야 한다.

또한, 원래 logging handler의 default format이 "%(message)s" 인데, `logstashHandler`는 포맷이 다르게 설정되어 있다. 그래서 formatter를 설정해주지 않으면 쓸데없는 포멧들이 같이 나온다. 이를 수정해 주었다.


[Tistory 원문보기](http://khanrc.tistory.com/66)
