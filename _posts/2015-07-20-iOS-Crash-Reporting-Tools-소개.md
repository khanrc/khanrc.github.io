---
layout: post
title: "iOS Crash Reporting Tools 소개"
tags: ['iOS']
date: 2015-07-20 18:06:00
---
# [Overview of iOS Crash Reporting Tools: Part 1/2](http://www.raywenderlich.com/33669/overview-of-ios-crash-reporting-tools-part-1)

iOS의 Crash Reporting에 대해 알아보자. 항상 요약번역한다고 하고 전체번역에 가까웠던거 같은데 이번엔 정말 요약번역에 도전한다!

원문이 2년 전(2013년 5월) 글이라서 조금 아쉽긴 한데 전체적으로 살펴보기에 이게 제일 좋을 것 같다. 참고로 파트 2에서는 각 툴을 설치하고 사용하는 방법을 소개하는데, 거기까진 필요없을 것 같다.

이 글에서는 전체적인 크래시 리포팅 툴들에 대해 알아본다.

## Introduction

이 시리즈에서는 크래시 리포팅 툴들을 살펴보고, 무료/유료 여부, 장점/단점 등을 알아본다. 첫번째 파트를 보고 나면 너에게 필요한 툴을 선택할 수 있을 것이고, 두번째 파트를 보고 나면 각 툴들을 어떻게 쓰는지 알 수 있을 것이다.

## Crash Reporting Basics

크래시 리포팅 툴은 두 요소로 구성된다: 리포팅 라이브러리 + 서버사이드 콜렉터. 당연히 둘 다 중요하다. 리포팅 라이브러리는 크래시를 대비하고(서버로 보내고), 서버사이드 콜렉터는 크래시 데이터를 모으고 유의미한 정보로 변환한다.

## Symbolication

크래시 리포트는 어플리케이션이 종료될 때 작동하던 모든 쓰레드의 스택 트레이스로 구성된다. 이 트레이스는 디버깅 할 때 볼 수 있는 정보와 비슷한데, 다만 심볼(인스턴스 및 메소드)의 이름에서 차이가 난다. 릴리즈 모드로 빌드하면 심볼 이름들은 바이너리 형태로 변환되고, 따라서 우리가 받는 크래시 리포트는 심볼 이름 대신에 16진수 주소가 적혀 있다.

iOS 디바이스를 맥에 연결하고, Xcode에서 Organizer를 열어서, 연결한 디바이스를 찾아 "Device Logs"를 선택하면 이러한 16진수 주소로 표현된 크래시 로그를 볼 수 있다:

![crash log](http://cdn2.raywenderlich.com/wp-content/uploads/2013/02/hexaCrash.png)

위에서 일부는 심볼이고 (e.g. `-[__NSArrayI objectAtIndex:]`) 일부는 16진수 주소 (e.g. `0x000db142 0xb1000 + 172354`) 인 것을 확인할 수 있다. 이는 partially symbolicated crash log라고 하는데, 크래시가 발생한 스택 트레이스의 꼭대기(top)에서 이 크래시를 생성한 구체적인 메소드까지 타고 내려간다.

이렇게 로그가 부분적으로 심볼화(partly symbolicated log) 되는 것은 Xcode가 UIKit이나 CoreFoundation같은 시스템 컴포넌트들만 심볼화 할 수 있기 때문이다(라인 6~18처럼). 

라인 3의 심볼화되지 않은(un-symbolicated) 부분인 `0x000db142 0xb1000 + 172354`가 무엇을 의미하는지 살펴보자. 0x000db142 = 0xb1000 + 172354 인데, 여기서 크래시가 발생했다는 뜻이다. 여기서 0xb1000은 앱의 시작 어드레스고, 172354는 시작 위치로부터의 현재 위치라고 할 수 있다.

그러나 이 raw crash report로는 디버깅을 할 수가 없으니, 이를 "symbolication"을 통해서 raw number들을 정확히 어떤 코드인가로 매칭하자.

## The Symbolication Process

심볼화를 위해서는 두가지가 필요하다: 크래시를 낸 어플리케이션 바이너리와 바이너리의 컴파일 과정에서 생성되는 dSYM 파일. Xcode에서 "Product/Archive"를 사용하면, Xcode가 이 파일들을 저장한다. Window/Organizer/Archive tab에서 확인할 수 있다. 네가 지금까지 빌드를 위해 "Build and Archive"를 사용했다면 지금까지의 정보도 저장되어 있을 것이다.

## Find that dSYM!

Window/Organizer/Archive tab에 가 보면 왼쪽에 우리가 아카이브한 앱들을 볼 수 있다. 앱을 오른쪽 클릭해서 show in finder옵션을 통해 파인더에서 열자. .xcarchive 파일(폴더)이 보일 것이다.

이 아카이브를 살펴보기 위해 터미널로 이동하자. 터미널에 이 폴더를 드래그하여 폴더 경로를 따고, Ctrl+A로 앞으로 이동한 뒤 cd를 쳐 주면 손쉽게 아카이브 폴더로 이동할 수 있다. 여기서 `find . -type d`명령을 통해 하위 폴더들을 살펴볼 수 있다.
    
    
    $ find . -type d
    .
    ./Products
    ./Products/Applications
    ./Products/Applications/breezi.app
    ./Products/Applications/breezi.app/_CodeSignature
    ./Products/Applications/breezi.app/en.lproj
    ./dSYMs
    ./dSYMs/breezi.app.dSYM
    ./dSYMs/breezi.app.dSYM/Contents
    ./dSYMs/breezi.app.dSYM/Contents/Resources
    ./dSYMs/breezi.app.dSYM/Contents/Resources/DWARF

"." 은 현재 폴더, "-type d"는 디렉토리를 의미한다. 즉 현재 위치로부터 하위 디렉토리들을 찾으라는 명령이다. 결과 중 "~.app.dSYM" 폴더가 우리가 심볼화를 위해 찾고자 하는 폴더다. 이 .dSYM이 바이너리로부터 심볼들을 복구할 수 있는 모든 정보를 갖고 있다.

Xcode는 자체적으로 심볼화를 지원한다. .dSYM파일을 찾기 위해 spotlight를 사용하는데, 가끔 파일이 컴퓨터에 없다던가 할 경우 문제가 생길 수 있다. 이럴 때 .dSYM파일을 찾아서 직접 심볼화 할 수 있다. Organizer/Devices tab/Device Logs로 가서 심볼화 할 크래시 로그를 선택하고 Re-Symbolicate 버튼을 누르면 된다.

![re-symbolicate](http://cdn3.raywenderlich.com/wp-content/uploads/2013/05/Resymbolicate-button.png)

Xcode 뿐만 아니라 크래시 리포팅 툴들도 전부 .dSYM파일을 사용한다. 이 파일이 없다면 크래시 로그를 갖고 있어도 아무 의미가 없다.

## Making a Case for Crash Reporting

.dSYM을 찾은 것 처럼, 투박하지만 크래시 리포팅 툴이 없을 때 수동으로 크래시 로그도 찾을 수 있다. 디바이스를 맥에 연결하고, OS별 크래시 로그 폴더를 찾은 뒤, ".crash" 확장자를 가진 파일을 찾는다. 이 파일을 가져와서 Xcode의 Organizer로 열면 "Device Logs" 탭에서 확인 및 심볼화(re-symbolication) 할 수 있다.

이 방법의 가장 큰 문제는 크래시가 난 디바이스가 있어야 한다는 것이다. 바꿔 말하면 유저가 직접 크래시 로그를 모아 너에게 보내줘야 한다! 

다행히도, 애플은 앱스토어를 통해 배포된 앱들이 크래시 리포트를 모을 수 있도록 해 준다. 단, 이 방법은 "진단 데이터를 자동으로 애플로 전송" 옵션에 동의한 유저에 한해서다. 앱스토어에 등록한 앱이 있다면, [아이튠즈 커넥트](http://itunesconnect.apple.com/)로 가서, "Manage Your Applications -&gt; 앱 선택 -&gt; View details -&gt; Crash Reports" 를 선택하자.

![crash report](http://cdn2.raywenderlich.com/wp-content/uploads/2013/02/appStore2-700x201.png)

여기서 애플이 모은 크래시 로그들을 볼 수 있다. 많은 사람들이 진단 데이터를 애플로 전송하는 데에 동의하지 않기 때문에 크래시 리포팅 툴을 사용하는 것에 비해서는 적은 크래시 로그들만이 확인 가능하다. 또한, 임포팅(importing) 및 심볼화(symbolicating) 등 추가 작업이 필요하다.

자, 그럼 지금까지 살펴본 작업들을 포괄적으로 제공하는 크래시 리포팅 툴들을 살펴볼 차례다. 위에서도 적었지만 원문이 2년 전 글이므로 많이 달라졌을 수 있다. 참고로 아래 툴 중 TestFlight는 애플이 인수했다!

  * Crashlytics
  * Crittercism
  * Bugsense
  * TestFlight
  * HockeyApp

## [Crashlytics](https://try.crashlytics.com/) (무료)

최근에 트위터에 인수된 Crashlytics는 가장 유명한 크래시 리포트 툴 중 하나다. 홈페이지에 가 보면 Path, Yammer, yelp, PayPal, Walmart 등등 짱짱한 회사들이 이걸 쓰고 있다. 클라이언트 / 서버사이드 프레임워크를 모두 지원하는 풀스택 서비스다. 원래 무료가 아니었던것 같은데 트위터가 인수하면서 무료로 바뀐 것 같다!

![crashlytics](http://cdn2.raywenderlich.com/wp-content/uploads/2013/02/crashlytics-700x376.png)

대시보드에서 잘 정리된 데이터를 볼 수 있다. 또한 크래시 리포트를 선택하면

![crash report](http://cdn2.raywenderlich.com/wp-content/uploads/2013/02/crashlytics2-e1361875571818.png)

이와 같이 심볼화 된 리포트를 볼 수 있다! 최상단에서는 크래시가 날 때의 환경이 나온다.

## [Crittercism](http://www.crittercism.com/)

[Pricing plan](https://www.crittercism.com/pricing). B2C/MAU 기준 30,000명 까지 무료. 프로 플랜은 더 많은 기능을 지원하고 유료. 한국어로 검색하면 거의 자료가 없는 걸 보니 한국에서 많이 쓰진 않는 듯하다. 

Crittercism은 또다른 풀스택 툴이다. Pinterest, adidas, NETFLIX, YAHOO, Linkedin, ebay 등이 사용한다. 심볼화된 크래시 로그와 대시보드를 살펴보자:

![dashboard](http://cdn4.raywenderlich.com/wp-content/uploads/2013/02/crittercism1-e1361877259704.png)

한눈에 잘 들어오진 않는다. Crittercism은 "breadcrumbs"이라는 기능을 지원하는데, 로그를 남겨 크래시가 나기 전에 앱이 어떤 컨텍스트에 있었는지 알 수 있게 해준다:

![breadcrumbs](http://cdn1.raywenderlich.com/wp-content/uploads/2013/02/crittercism2-700x359.png)

> 단, breadcrumbs는 유료 enterprise 계정에서 지원하는 기능이다.

또한 Crittercism은 크래시 로그가 어느 지역에서 발생하였는지 볼 수 있는 기능을 지원한다:

![mapping](http://cdn3.raywenderlich.com/wp-content/uploads/2013/02/crittercism3-700x334.png)

단 이 기능은 유저에게 위치정보 허가를 요청하지 않는다. 사용에 있어 조심하자.

Crittercism에는 독특한 가격 플랜이 있는데 앱이 일반 사용자를 위한 앱이냐 아니면 기업을 위한 앱이냐(B2C or B2B) 에 따라 가격 플랜이 달라진다. 

Crittercism은 다양한 플랫폼을 지원하지만 사용의 편의성에 있어 조금 떨어진다.

## [Bugsense](https://mint.splunk.com/)

Bugsense를 검색하면 Bugsense는 안 나오고 splunk mint라는 게 나온다. splunk에 인수된 건지, 이름을 바꾼 건지는 잘 모르겠지만 아무튼 현재는 splunk mint 라는 이름으로 서비스하고 있다. 재미있는 점은 splunk korea가 있다는 점이며, splunk로 검색하면 꽤 많은 자료가 나오는 것으로 보아 꽤 많이 쓰이고 있는 것 같다. splunk는 통합적인 데이터 관리 툴 및 플랫폼을 서비스하며, splunk mint는 그중에서도 모바일 툴이다.

[Pricing plan](http://www.splunk.com/en_us/products/pricing.html). splunk mint 또한 MAU에 기반하여 가격이 책정되는데, 자세한 건 홈페이지에도 나와 있지 않고 [직접 문의](http://www.splunk.com/en_us/talk-to-sales.html?expertCode=sales)하라고 적혀 있다 -_-

아무튼 원문은 Bugsense기준으로 작성되었으니 이하 Bugsense로 작성하도록 한다.

Bugsense도 풀스택 서비스를 지원하고, 삼성, 인텔, 그루폰 등의 대기업에서 사용한다(다만 그루폰은 이것저것 다 쓰는 건지 어딜 가나 고객 목록에 그루폰이 있다). Crittercism처럼 다양한 플랫폼을 지원한다.

초기 셋업은 다른 플랫폼과 유사하게 계정을 만들고, SDK를 받아서, 프로젝트에 설치하고 API key를 세팅하면 된다. 대시보드는 이렇게 생겼다:

![dashboard](http://cdn4.raywenderlich.com/wp-content/uploads/2013/05/bug_sense_dashboard-700x377.png)

로그를 선택하면 자세히 볼 수 있다:

![crash log](http://cdn3.raywenderlich.com/wp-content/uploads/2013/05/crash-detail-700x385.png)

어느 함수에서 어떤 에러를 발생시켰는지 정확한 코드 위치와 함께 볼 수 있다.

Bugsense의 한 가지 단점은, 서버사이드 심볼화를 하기 위해선 수동으로 dSYM파일을 업로드 해야 한다는 점이다. 클라이언트에서 심볼화를 할 수도 있지만, 이렇게 하면 크래시 코드 라인과 같은 세부 정보를 알 수 없다.

Bugsense 또한 어플리케이션의 컨텍스트를 추적할 수 있는 breadcrumbs를 제공하며, 버그를 수정했을 경우 해당 버그를 발생시킨 유저에게 푸쉬를 보낼 수 있는 "Fix notifications" 서비스 또한 재밌는 기능이다.

## [TestFlight](http://testflightapp.com) (무료)

위 링크를 눌러보면 애플 홈페이지로 이동한다! 애플에 인수되었다. 당연히, 공짜다.

TestFlight는 베타버전의 배포를 위해 탄생했다. 버전이 올라가면서 액션 로깅이나 크래시 리포팅 기능 등이 추가되었다. TestFlight는 Adobe, Instagram, tumblr 등이 배포(over-the-air deployment), 트래킹, 크래시 리포팅을 위해 사용한다. 애플에 인수되었다길래 iOS 용만 있는 줄 알았는데 Android 용도 있다고 한다.

".ipa" 파일을 서버에 업로드해서 배포할 수 있다. 서버사이드 대시보드는 아래와 같이 생겼다:

![dashboard](http://cdn2.raywenderlich.com/wp-content/uploads/2013/02/testflight-700x439.png)

대시보드의 왼쪽 메뉴에서 세션, 유저 피드백, 체크포인트 크래시(breadcrumbs와 유사) 등을 확인할 수 있다.

![crash report](http://cdn4.raywenderlich.com/wp-content/uploads/2013/02/tf-700x332.png)

위에서 언급했다시피, 유저의 액션을 로깅할 수 있다. (원문 글 작성 시점인 2년 전에는) 써드파티 앱과의 연동이 불가능하다. 반면 베타 버전 배포와 크래시 리포트, 액션 로깅 등이 하나로 전부 가능하다는 점은 매력적이다.

## HockeyApp

14년 12월에 마이크로소프트가 인수. Crashlytics와는 달리 그렇다고 해서 무료로 전환되진 않았다. [Pricing plan](http://hockeyapp.net/pricing/). 비지니스 플랜으론 3 Owners / 15 Apps가 매달 30달러, 퍼스널 플랜으론 5 Apps가 매달 10 달러.

HockeyApp은 인디 개발자 세계에서는 잘 알려져 있다(인디 개발자가 만들었다고 한다). TestFlight처럼, 크래시 리포팅 외에도 베타버전 배포 관리 기능도 제공한다.

HockeyApp 셋업은 다른 프레임웤들과 동일하지만, 전체 소스를 임베딩하는 방법도 있다. 혹시 버그가 있을 경우에 직접 고치는 게 가능하다! (HockeyApp은 Quincykit이라는 오픈소스의 호스팅 버전인 듯 하다)

![dashboard](http://cdn1.raywenderlich.com/wp-content/uploads/2013/02/hockeyapp-e1361882100809.png)

대시보드 상단에 메뉴가 있고, 버전별 정보를 요약해서 보여주고 하단에는 통계 그래프가 있다. 데스크탑 앱은 Xcode에서 아카이브가 끝나면 새로운 빌드와 dSYM을 업로드한다. 한 번 직접 dSYM 파일을 서버에 업로드하면, Crashes tab을 선택할 수 있고 로그를 눌러 심볼화된 크래시 로그를 볼 수 있다:

![crash log](http://cdn2.raywenderlich.com/wp-content/uploads/2013/02/hockeyapp2-700x440.png)

HockeyApp 백엔드는 로그 서치 기능을 제공하는데, "show all of the crashes that happened on iOS6 but not on an iPad" 과 같은 걸 할 수 있다.

만약 직접 서버사이드 호스팅이 가능하다면, [QuincyKit](http://quincykit.net/)을 고려하자. HockeyApp은 QuincyKit을 호스팅하는 버전이다.

## Summary and Comparison Chart

![summary and comparison](http://cdn2.raywenderlich.com/wp-content/uploads/2013/05/table.png)

2년 전 버전이니 참고만 하자. 

## The Bottom Line (최종 결과)

원문 저자의 의견으로는, Crashlytics가 최고의 크래시 리포팅 툴이다. 공짜이고, 사용하기에 편리하며, 리포팅 프로세스가 전부 자동화 되어 있어 dSYM파일을 직접 업로드 할 필요가 없다. 단점이 있다면 앱 배포 기능을 제공하지 않는다는 점이다.

만약 크로스 플랫폼 서비스를 찾는다면, Bugsense (splunk mint) 를 추천한다. 대쉬보드가 훌륭하다. 단, 저렴한 티어에서는 데이터가 오래 저장되지 않는다 (7일에서 30일 정도).

## In my case

아마 Crashlytics를 기본적으로 사용하고 TestFlight를 고려할 것 같다. 어차피 베타버전 배포 및 유저 행동 로깅 또한 필요하기 때문에 Crashlytics만으론 해결이 안 된다. TestFlight를 좀 더 알아보고 결정할 예정이다.

Getting started에 해당하는 [파트 2](http://www.raywenderlich.com/34050/overview-of-ios-crash-reporting-tools-part-2)는 따로 번역하지 않을 예정이다. 필요하면 원문을 참고하자. 어떤 툴을 사용할 지 결정했다면 해당 툴로 검색하여 사용법을 찾아보는 것이 낫지 않을까 싶다.

[TestFlight vs. Crashlytics vs. HockeyApp](https://github.com/lapwinglabs/blog/blob/master/bluu-testflight-crashlytics-hockeyapp.md) 에 따르면, 크래시 리포팅에는 Crashlytics가 좋고, 베타버전 배포에는 TestFlight가 좋아서 얘네는 둘 다 쓰기로 했다고 한다(Crashlytics도 베타버전 배포를 지원하는 모양이다). 나도 그런 방향으로 진행하기로 결정했다. 아, 저 글에서 HockeyApp은 유료라서 바로 비교에서 배제하고 있다 -_-;


[Tistory 원문보기](http://khanrc.tistory.com/108)
