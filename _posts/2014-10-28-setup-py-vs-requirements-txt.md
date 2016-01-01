---
layout: post
title: "setup.py vs requirements.txt"
tags: ['Python']
date: 2014-10-28 11:00:00
---
# [setup.py vs requirements.txt](https://caremad.io/blog/setup-vs-requirement/)

이 포스트는 위 링크의 번역 + 요약 + @ 이다.

`setup.py`와 `requirements.txt`간에는 수많은 오해들이 있다. 많은 사람들이 이 두가지가 중복된다고 생각한다.

## Python Libraries

이 포스트에서 파이썬 라이브러리란 이미 디플로이되고 릴리즈되어 공개된 라이브러리를 의미한다. [PyPI](https://pypi.python.org/pypi)에서 이러한 라이브러리들을 찾아볼 수 있다. 라이브러리는 제대로 배포되기 위해서 여러 [메타데이터](http://ko.wikipedia.org/wiki/%EB%A9%94%ED%83%80%EB%8D%B0%EC%9D%B4%ED%84%B0)를 가지고 있다. 이름이라던가, 버전, 디펜던시 등이 바로 그것이다. `setup.py`는 바로 이런 메타데이터에 대한 명세다:
    
    
    from setuptools import setup
    
    setup(
        name="MyLibrary",
        version="1.0",
        install_requires=[
            "requests",
            "bcrypt",
        ],
        # ...
    )
    

헌데, 위 명세에서 제공하는 디펜던시에는 라이브러리를 가져올 url이라던가, 각 라이브러리의 버전 등이 명시되어 있지 않다. 이는 굉장히 중요한 정보이며 따라서 위와 같은 디펜던시 명세는 `abstract dependencies`라 부른다(정확하게는 글의 저자가 그렇게 부른다).

## Python Applications

여기서 파이썬 어플리케이션이란 네가 deploy할 그것이다. 그건 PyPI에 존재할 수도 있고 아닐수도 있지만, 재사용할만한게 많지는 않은 그런 것이다. PyPI에 존재하지 않는 어플리케이션은 명확한 deploy config file이 필요하다. 이 섹션에서는 바로 그 _deploy specific_한 부분을 다룬다.

일반적으로 어플리케이션은 디펜던시들을 가지고, 종종 이 디펜던시는 굉장히 복잡하다. 디플로이된 특정한 인스턴스들은, 보통 이름도 없고 어떤 다른 패키징 메타데이터가 전혀 없다. 이는 pip requirement file에 반영된다. requirement file은 보통 이렇게 생겼다:
    
    
    # This is an implicit value, here for clarity
    --index-url https://pypi.python.org/simple/
    
    MyPackage==1.0
    requests==1.2.0
    bcrypt==1.0.2
    

이렇게 각 정확한 버전과 함께 디펜던시 명세를 작성한다. 라이브러리는 최신 버전을 사용하는 경향이 있는 반면 어플리케이션은 특정한 버전을 필요로 하는 경향이 있다. 예를 들어 `requests`의 어떤 버전을 사용하는지는 중요하지 않지만 production에서 사용하는 버전과 development에서 사용하는 버전은 같아야 한다.

위 파일에서 _--index-url <https://pypi.python.org/simple/>_를 볼 수 있다. PyPI를 사용하는 일반적인 `requirements.txt`에서는 이렇게 명확하게 표시하지 않는다. 그러나 이건 `requirements.txt`에서 중요한 부분이다. 이 한 줄이 `abstract dependency`를 `concrete dependency`로 바꾼다. 

## 그래서, Abstract건 Concrete건 무슨 상관인데?

이 차이 - abstract냐 concrete냐 - 는 매우 중요하다. 이것은 회사가 PyPI같은 형태의 프라이빗 패키지 인덱스를 구축할 수 있게 해준다. 심지어 네가 라이브러리를 `fork`했다면 이 라이브러리를 사용할 수 있게도 해준다. `abstract dependency`는 오직 이름과 버전만 사용하기 때문에 너는 PyPI를 통해서나, 또는 Crate.io를 통해서나, 또는 너의 파일시스템으로부터 인스톨할 수 있다. 나아가 알맞은 이름과 버전만 명시한다면 라이브러리를 포크하고, 코드를 변경하여도 문제없이 사용할 수 있다.

`abstract requirement`를 사용해야 하는 곳에 `concrete requirement`를 사용할 때 나타나는 극단적인 문제는 `Go`에서 찾아볼 수 있다. `Go`에서는 `import`에서 아래와 같이 url을 사용할 수 있다:
    
    
    import (
        "github.com/foo/bar"
    )
    

보다시피 디펜던시를 위해 정확한 url을 사용할 수 있다. 이 상황에서, bar라이브러리에 버그가 있어 이를 포크하여 수정하여 사용한다면, bar만 포크하면 되는 것이 아니라 bar에 관련된 모든 라이브러리를 포크해야 한다. 단지 bar의 작은 변경을 위해서.

## A Setuptools Misfeature

`Setuptools`에도 `Go`와 비슷한 특징이 있다. `dependency_links`라는 것이다:
    
    
    from setuptools import setup
    
    setup(
        # ...
        dependency_links = [
            "http://packages.example.com/snapshots/",
            "http://example2.com/p/bar-1.0.tar.gz",
        ],
    )
    

이 `setuptools`의 특징은 이 디펜던시의 `abstractness`를 파괴한다. 이제 여기에도 위 `Go`의 문제와 같은 문제가 발생한다.

## Developing Reusable Things or How Not to Repeat Yourself

라이브러리와 어플리케이션의 차이는 분명하지만, 네가 라이브러리를 개발한다면 그건 네 어플리케이션의 일부일 것이다. 특정한 로케이션에서, 특정한 디펜던시를 원한다면 `setup.py`에 `abstract dependency`를, `requirements.txt`에 `concrete dependency`를 넣어야 한다. 그러나 이 두 분리된 리스트를 관리하는게 싫다면 어떻게 해야 할까? `requirements.txt`에서는 이러한 케이스를 위한 기능을 지원한다. `setup.py`가 있는 디렉토리에 아래와 같이 `requirements.txt`를 만들 수 있다:
    
    
    --index-url https://pypi.python.org/simple/
    
    -e .
    

이제, `pip install -r requirements.txt`를 하면 이전과 동일하게 작동한다. file path . 에 있는 `setup.py`를 찾아, 거기에 있는 `abstract dependency`를 requirement파일의 `--index-url`과 결합시켜 `concrete dependency`로 바꿔 인스톨한다.

이 방식은 강력한 장점을 갖는다. 네가 조금씩 개발하고 있는 여러 라이브러리가 있다고 하자. 또는 하나의 라이브러리를 여러 부분으로 잘라 사용하고 있다고 하자. 어떻든 그 라이브러리의 공식 릴리즈 버전과는 다른 development version을 사용하고 있는 것이다. 그렇다면 `requirements.txt`를 이렇게 쓸 수 있다:
    
    
    --index-url https://pypi.python.org/simple/
    
    -e https://github.com/foo/bar.git#egg=bar
    -e .
    

먼저 명시된 url인 github 으로부터 bar 라이브러리를 설치한다. 그리고 나서 `--index-url`로부터 나머지 라이브러리들을 설치한다. 이 때 bar에 대한 디펜던시는 이미 해결되었기 때문에 추가적인 인스톨을 하지 않는다. 즉, bar라이브러리에 대한 development version을 사용할 수 있는 것이다.


[Tistory 원문보기](http://khanrc.tistory.com/61)
