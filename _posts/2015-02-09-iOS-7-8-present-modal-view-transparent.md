---
layout: post
title: "iOS 7/8 present modal view transparent"
tags: ['iOS']
date: 2015-02-09 16:52:00
---
iOS 7, 8에서 modal view controller를 (모달이라는 용어는 iOS 6부터였나 함수명에서 사라졌는데 왜인진 모르겠다. 아무튼 모달이라고 부르자) 뒤가 보이게 띄우는 방법이다. iOS 7에서는 테스트해보지 않았지만 iOS 8에선 잘 되는 걸 확인.

참고로, iOS 8에서 present modal view controller에서 살짝 랙이 걸리는데, 버그인 것 같다. dispatch로 delay를 걸어 0초 후에 띄워주면 랙 없이 뜬다.

  


ios7:  
presentingVC.modalPresentationStyle = UIModalPresentationCurrentContext;  
  
  
ios8:  
modalVC.modalPresentationStyle = UIModalPresentationOverCurrentContext;  
modalVC.modalTransitionStyle = UIModalTransitionStyleCrossDissolve;  
  
and then in both:  
[presentingVC presentViewController:modalVC animated:YES completion:nil];

  


  


출처: http://www.raywenderlich.com/forums/viewtopic.php?f=2&amp;t=18661


[Tistory 원문보기](http://khanrc.tistory.com/83)
