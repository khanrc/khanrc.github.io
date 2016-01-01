---
layout: post
title: "UIScrollView touch event"
tags: ['iOS']
date: 2015-02-15 00:31:00
---
예전에는, 스크롤뷰로부터 터치 이벤트를 캐치하기 위해 스크롤뷰를 상속받아서 touches- 메소드들을 오버라이딩 해 주어야 했다. 스크롤뷰가 터치 이벤트를 죄다 먹어버리기 때문이었다.

당시에 그 작업을 했던 게 iOS 4.3 을 지원하던 한 2-3년 전쯤이었던거 같다.

  


이번에 이 작업을 다시 해 보니, touches- 메소드가 터치 이벤트를 제대로 받아오지 못한다. 자세히 찾아보진 않았지만 스크롤뷰에서 터치 이벤트를 가져오는 방식이 touches- 메소드에서 제스처 기반으로 바뀐 것 같다. 이제는 터치 이벤트를 가져오려면 스크롤뷰를 오버라이딩 하지 말고 제스처를 붙이고, 동시에 여러 제스처를 인식하도록 델리게이트를 설정해주면 된다.

  


참고: http://stackoverflow.com/questions/13736399/intercepting-pan-gestures-over-a-uiscrollview-breaks-scrolling

  


p.s: https://www.cocoanetics.com/2010/06/hacking-uiscrollview-gesture-recognizers/ 이런걸 보면 예전에 바뀐 거 같은데 왜 내가 전에 작업할땐 그랬고 최근 스택오버플로 답변들도 스크롤뷰를 상속받으라고 하는지는 잘 모르겠다. 알게 되면 추가함.


[Tistory 원문보기](http://khanrc.tistory.com/84)
