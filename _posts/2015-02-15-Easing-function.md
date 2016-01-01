---
layout: post
title: "Easing function"
tags: ['iOS']
date: 2015-02-15 11:50:00
---
# Easing function

**Easing functions** specify the rate of change of a parameter over time.  
즉, 시간에 따른 속도의 변화를 나타내는 함수다. 

썩 유명한 개념인 것 같은데, 나만 몰랐던 거 같다 -_-;  
애니메이션을 만들고자 한다면 필수적으로 들어가는 개념이니 이번 기회에 잘 알아두도록 하자.

  * Ease In: 속도가 점점 증가
  * Ease Out: 속도가 점점 감소
  * Ease InOut: 속도가 점점 증가하다 다시 감소

## [Bezier curve](http://ko.wikipedia.org/wiki/%EB%B2%A0%EC%A7%80%EC%97%90_%EA%B3%A1%EC%84%A0)

베지에 곡선. n개의 점으로부터 얻어지는 n-1차 곡선이다. 즉, 시작점과 끝점이 있고 중간에 2개의 조절점이 있다면 3차 베지에 곡선이다. 워드에서 곡선을 그릴 때 느낌인 듯. 아래 구성 애니메이션을 참고하자:  
<http://www.jasondavies.com/animated-bezier/>  
<http://en.wikipedia.org/wiki/Bezier_Curve#Constructing_B.C3.A9zier_curves>

![quadratic bezier curve](http://upload.wikimedia.org/wikipedia/commons/d/db/B%C3%A9zier_3_big.gif)

Easing function은 이 베지에 곡선으로 만들어진다. [애플의 프로그래밍 가이드](https://developer.apple.com/library/ios/documentation/Cocoa/Conceptual/Animation_Types_Timing/Articles/Timing.html)를 보면:
    
    
    CAMediaTimingFunction *customTimingFunction;
    customTimingFunction=[CAMediaTimingFunction functionWithControlPoints:0.25f :0.1f :0.25f :1.0f];
    

이렇게 코어 애니메이션에서 시작점과 끝점 (0, 0), (1.0, 1.0)외에 두개의 조절점 (0.25, 0.1), (0.25, 0.1)을 추가하여 베지에 곡선을 만든다. 코어 애니메이션으로 Easing function을 적용하는 예제는 여길 참고하자: <http://lafosca.cat/smooth-and-custom-animations-in-ios/>

## Libraries

  * [AHEasing](https://github.com/warrenm/AHEasing): 가장 기본이 되는 Easing function library. 수학적 함수들과 그 래퍼로 구성됨.
  * [UIView+EasingFunctions](https://github.com/zrxq/UIView-EasingFunctions): 써보진 않았는데, UIView animation에서 AHEasing의 function들을 사용할 수 있도록 만들어 놓은 extension
  * [SCScrollView](https://github.com/stefanceriu/SCScrollView): 스크롤뷰에서 contentOffset을 조정할 때 easing function을 사용할 수 있도록 한 라이브러리

## 참고

  * [Easing Functions Cheat Sheet](http://easings.net/)
  * [Easing Equation](http://gizma.com/easing/)
  * [Cubic-bezier](http://cubic-bezier.com/): 몇가지 베지에 곡선을 살펴볼 수 있다


[Tistory 원문보기](http://khanrc.tistory.com/85)
