---
layout: post
title: "HitTesting"
tags: ['iOS']
date: 2015-08-05 18:40:00
---
# Hit-Testing in iOS

Hit-testing이란, 유저 인터랙션, 즉 hit이 들어왔을 때 이 hit을 받는 뷰를 판단하는 과정이다. 뷰들이 쌓인 관계는 트리 구조로 구성되어 있는데, 이 트리에서 루트부터 시작해서 reverse pre-order depth-first traversal 알고리즘을 통해 hit을 받는 노드(뷰)를 찾아낸다.

Hit-testing은 모든 터치(유저 인터렉션)마다 실행되며 그 결과로 어떤 뷰나 gesture recognizer가 해당 터치에 대한 정보를 담고 있는 `UIEvent`를 받는다.

> Note: 이유는 모르지만 hit-testing은 연속적으로 여러번 불린다.

hit-testing이 끝나면 해당 터치를 받는 뷰가 hit-test view가 되고, 이 뷰는 모든 터치 이벤트(began, moved, ended, canceled 등)를 받는 `UITouch`오브젝트와 연동된다. 또한 이 뷰나 상위 뷰에 gesture recognizer가 붙어 있으면 마찬가지로 `UITouch`오브젝트와 연동된다. 그리고 나면 이제 hit-test view는 터치 이벤트들을 받기 시작한다.

중요한 점은 유저 인터렉션이 해당 뷰 밖으로 나가도 hit-test view는 변하지 않는다.

![search](http://smnh.me/images/hit-test-depth-first-traversal.png)

위 이미지는 터치가 되었을 때 view hierarchy 트리에서 hit-testing 과정을 보여주고 있다. 형제(sibling)들의 경우 역순(reverse)으로 검색함으로써 겹치는 부분이 있을 때는 더 오른쪽에 있는 나중에 추가한 뷰가 hit-test view로 선택된다.

이 알고리즘은 다음과 같이 코딩할 수 있다. 아래는 `hitTest:withEvent:` 메소드의 기본 구현체(implementation)다.
    
    
    - (UIView *)hitTest:(CGPoint)point withEvent:(UIEvent *)event {
        if (!self.isUserInteractionEnabled || self.isHidden || self.alpha <= 0.01) {
            return nil;
        }
        if ([self pointInside:point withEvent:event]) {
            for (UIView *subview in [self.subviews reverseObjectEnumerator]) {
                CGPoint convertedPoint = [subview convertPoint:point fromView:self];
                UIView *hitTestView = [subview hitTest:convertedPoint withEvent:event];
                if (hitTestView) {
                    return hitTestView;
                }
            }
            return self;
        }
        return nil;
    }

  * `alpha`가 0.01 이하인 경우 hidden으로 취급한다
  * `pointInside:withEvent:` 함수는 뷰 안에 해당 포인트가 포함되는지를 검사한다
  * hit-test view를 재귀적으로 찾아 리턴하고, 만약 없으면 nil을 리턴한다

코드가 어렵지 않으니 자세한 설명은 생략한다.

### Common use cases for overriding `hitTest:withEvent:`

`hitTest:withEvent:`를 오버라이딩 하면 터치 이벤트를 받는 hit-test view를 변경할 수 있다.

#### Increasing view touch area

버튼이나 뷰에 탭 제스처를 붙였을 때, 해당 버튼의 크기보다 좀 더 큰 영역의 터치를 받아야 할 때가 자주 있다.   
![increase touch area](http://smnh.me/images/hit-test-increase-touch-area.png)
    
    
    - (UIView *)hitTest:(CGPoint)point withEvent:(UIEvent *)event {
        if (!self.isUserInteractionEnabled || self.isHidden || self.alpha <= 0.01) {
            return nil;
        }
        CGRect touchRect = CGRectInset(self.bounds, -10, -10);
        if (CGRectContainsPoint(touchRect, point)) {
            for (UIView *subview in [self.subviews reverseObjectEnumerator]) {
                CGPoint convertedPoint = [subview convertPoint:point fromView:self];
                UIView *hitTestView = [subview hitTest:convertedPoint withEvent:event];
                if (hitTestView) {
                    return hitTestView;
                }
            }
            return self;
        }
        return nil;
    }

위 소스는 inset을 상하좌우 -10씩 주어 터치 범위를 넓혀준다. 기본 구현체와 비교했을 때 `pointInside:withEvent:`가 `CGRectContainsPoint()`로 바뀐 것을 확인할 수 있다. 이를 이용하면 `hitTest:withEvent:`를 수정하지 않고 `pointInside:withEvent:`를 오버라이드하여 같은 효과를 구현할 수 있다.
    
    
    - (BOOL)pointInside:(CGPoint)point withEvent:(UIEvent *)event {
        CGRect touchRect = CGRectInset(self.bounds, -10, -10);
        return CGRectContainsPoint(touchRect, point);
    }

참고로, 두 경우 모두 이 뷰의 수퍼뷰가 확장 범위를 포함하여야 한다. 다시 말해 이 뷰의 터치 범위를 상하좌우 -10씩 확장했을 때 수퍼뷰가 그 범위를 포함하고 있어야 한다. 그렇지 않다면 수퍼뷰도 마찬가지로 `hitTest:withEvent:`를 오버라이딩 해 주어야 한다. 트리 구조로 위에서 타고 내려오므로 부모 노드에서 더이상 내려오지 않으면 아무리 자식 노드의 인식 범위를 확장한들 소용이 없다.

> `pointInside:withEvent:`의 오버라이딩은 <http://stackoverflow.com/a/13067285>를 참고하였다.

#### Passing touch events through to views below

때로는 터치 이벤트를 무시하고 아래 뷰로 넘겨야 할 때가 있다. 예를 들면 전면에 알파값을 준 투명한 뷰로 덮는 경우. 이러한 뷰에서 터치 이벤트를 받지 않고 통과시켜 뒤에 있는 뷰가 동작할 수 있도록 할 수 있다.
    
    
    - (UIView *)hitTest:(CGPoint)point withEvent:(UIEvent *)event {
        UIView *hitTestView = [super hitTest:point withEvent:event];
        if (hitTestView == self) {
            hitTestView = nil;
        }
        return hitTestView;
    }

> 다만 `view.userInteractionEnabled = NO;` 와 무엇이 다른지는 잘 모르겠음

#### Passing touch events to subview

부모 뷰가 여러 자식 뷰 중 특정 자식 뷰를 지정해서 터치 이벤트를 넘기고 싶을 수 있다. 예를 들어, 이미지 캐러셀(carousel)을 생각해보자. 

![carousel](http://smnh.me/images/hit-test-pass-touches-to-subviews.png)

페이징을 위해 `UIScrollView`를 작은 크기(페이지 크기)로 만들고, 보이기는 다 보여야 하니까 `clipsToBounds`를 `NO`로 설정한다. 이렇게 하고 나면 캐러셀이 잘 작동하지만 페이징을 위해 작게 만들어 놓은 `UIScrollView`때문에 해당 부분을 스크롤해야만 이미지가 넘어가는 문제가 있다. 이 문제를 해결하기 위해 부모뷰 어디를 터치해도 `UIScrollView`가 터치 이벤트를 받도록 오버라이딩 하자.
    
    
    - (UIView *)hitTest:(CGPoint)point withEvent:(UIEvent *)event {
        UIView *hitTestView = [super hitTest:point withEvent:event];
        if (hitTestView) {
            hitTestView = self.scrollView;
        }
        return hitTestView;
    }

## 참고

[Hit-Testing in iOS](http://smnh.me/hit-testing-in-ios/)   
[stackoverflow; Event handling for iOS - how hitTest:withEvent: and pointInside:withEvent: are related?](http://stackoverflow.com/questions/4961386/event-handling-for-ios-how-hittestwithevent-and-pointinsidewithevent-are-r)   
[stackoverflow; UIButton: Making the hit area larger than the default hit area](http://stackoverflow.com/a/13067285)


[Tistory 원문보기](http://khanrc.tistory.com/114)
