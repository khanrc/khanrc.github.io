---
layout: post
title: "CALayer: anchorPoint"
tags: ['iOS']
date: 2015-02-21 00:02:00
---
# CALayer: anchorPoint

CA는 Core Animation. 이 글은 스택오버플로 <http://stackoverflow.com/questions/12208361/why-does-my-view-move-when-i-set-its-frame-after-changing-its-anchorpoint/12208587#12208587> 답변의 번역이다.

## answer

`CALayer`가 수퍼레이어로부터 어떻게 나타나는지를 결정하는 네가지 프로퍼티가 있다:

  * `position` (`anchorPoint`의 포지션)
  * `bounds` (정확히는 `bounds`의 `size`파트만)
  * `anchorPoint`
  * `transform`

보다시피 `frame`은 이 프로퍼티에 포함되지 않고, 위 프로퍼티로들로부터 만들어진다.

(100, 100, 100, 20)의 프레임을 갖는 `UILabel`을 만들었다고 하자. `UILabel`은 `UIView`의 서브클래스이며 모든 `UIView`는 레이어를 갖는다. 이 레이어는 디폴트 앵커 포인트로 (0.5, 0.5)를 가진다. 따라서 `bounds`는 (0, 0, 100, 20)이 되고 `position`은 (150, 110)이 된다. 

![1](http://i.stack.imgur.com/E3KSK.png)

여기서 `anchorPoint`를 (1,1)로 바꾸면 `position`은 그대로기 때문에 아래와 같이 변한다.

![2](http://i.stack.imgur.com/4izAN.png)

## apply

<http://stackoverflow.com/questions/14007983/scale-uiview-with-the-top-center-as-the-anchor-point>

앵커 포인트를 조정하여 뷰의 중심을 고정시켜 놓고 scale animation을 적용할 수 있다.  
아래는 뷰를 상단 중앙에 고정시키는 소스다:
    
    
    CGRect frame = view.frame;
    CGPoint topCenter = CGPointMake(CGRectGetMidX(frame), CGRectGetMinY(frame));
    
    view.layer.anchorPoint = CGPointMake(0.5, 0);
    view.layer.position = topCenter;
    

이후 scale 애니메이션을 적용하면 상단 중앙에 고정된 것을 확인할 수 있다.


[Tistory 원문보기](http://khanrc.tistory.com/86)
