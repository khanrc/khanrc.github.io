---
layout: post
title: "SharedCount: 소셜 인기도 확인하기"
tags: ['Server(Back-end)']
date: 2014-09-15 23:31:00
---
# SharedCount: Facebook, Twitter 공유 개수 확인하기

[SharedCount.com](http://www.sharedcount.com/)이라는 서비스가 있다.  
들어가보면 바로 확인할 수 있는데, URL을 넣으면 Facebook, Twitter, Google+, Pinterest, LinkedIn 등에서 이 URL이 얼마나 공유되었는지 알려준다.

Pricing에 가 보면 가격 정책을 볼 수 있다. 공짜로는 하루에 1만 번까지. 한달에 40달러만 내면 매일 20만번의 요청이 가능하니, 귀찮으면 이 서비스를 사용하는 것도 나쁘지 않다.

내가 하는 프로젝트에서는 Facebook과 Twitter만 체크하기 때문에, SharedCount를 사용하기보단 직접 만들기로 했다.

<http://www.sharedcount.com/documentation.php> 에 보면 SharedCount가 공유 데이터를 어떤 API를 사용해서 가져오는지 상세하게 잘 나와 있다.

## Facebook

[https://api.facebook.com/method/links.getStats?urls=%%URL%%&amp;format=json](https://api.facebook.com/method/links.getStats?urls=%%URL%%&format=json)  
SharedCount에서 사용하는 URL인데, 이게 최신이 아닌 걸로 보인다. 

<http://www.local-pc-guy.com/web-dev/getting-number-facebook-shares-likes-url>  
이 글을 참조하자. 근데 이 글에서 제시하는 fql도 deprecated 되었다 -_-

<https://developers.facebook.com/docs/graph-api/reference/v2.1/link>  
이게 최신의 graph-api다.

지금은 다 잘 작동하는걸로 보인다.

## on Python

urllib를 이용해서 간단하게 구현했다.  
딱히 api key도 필요 없는 것 같고, 별로 어려울 거 없지만 한 가지 짚고 넘어간다면  
urlencoding을 해주어야 한다.
    
    
    params = urllib.urlencode({"url": link_url})
    url = "http://urls.api.twitter.com/1/urls/count.json?" + params
    

이런 식으로.

참고로 urlencode에 다른 여러가지 키밸류를 넣으면 &amp;로 묶어 준다.


[Tistory 원문보기](http://khanrc.tistory.com/35)
