---
layout: post
title: "Auto Layout Tutorial"
tags: ['iOS']
date: 2015-01-15 17:16:00
---
# Beginning Auto Layout Tutorial in Swift: Part 1/2

> 여전히, 요약번역. 옵젝씨 코드는 [오리지널 포스트](http://www.raywenderlich.com/50317/beginning-auto-layout-tutorial-in-ios-7-part-1)를 참고하자.

오토레이아웃은 Xcode 4에 등장했지만 그 이후로 많이 발전했다. 이전에 시도하다 포기한 적이 있었다면 지금이 바로 다시 도전할 타이밍이다. Xcode 5와 iOS 7에서 나아졌고 현재 Xcode 6과 iOS 8에서는 다양한 스크린 사이즈로 인해 필수적이다.

또한 다양한 스크린 사이즈를 지원하는 것 외에도 국제화를 쉽게 만들어준다. 더이상 다양한 언어에 맞춰 새로운 xib를 만들 필요가 없다. Hebrew나 Arabic같이, 오른쪽에서 왼쪽으로 쓰는 언어들까지 포함해서.

자 그럼 시작해보자!

## The problem with springs and struts

Single View Application 템플릿으로 새 프로젝트를 만들자. 앱 이름은 "StrutsProblem"이라 하자.

![new project](http://cdn3.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate1-480x304.png)

> 나는 Obj-C로 진행하지만 본 튜토리얼은 원문이 스위프트로 되어 있으므로 스위프트로 진행한다.

"springs and struts" 모델로 알려진 **autosizing mask**를 잘 알고 있을 거다. 오토사이징 마스크는 수퍼뷰의 사이즈가 변할 때 그 자식뷰가 어떻게 변할지를 결정한다. 수퍼뷰가 flexible margin이나 fixed margin(the struts)을 갖고 있을 때, 자식뷰의 width와 height는 어떻게 되는가(the springs)?

만약 flexible width라면, 수퍼뷰가 커지면 자식뷰도 커진다. fixed right margin이라면, 자식뷰의 오른쪽 엣지가 수퍼뷰의 오른쪽 엣지와 일정한 마진을 유지한다.

이러한 오토사이징 시스템은 간단한 케이스에는 괜찮지만 조금만 복잡해져도 문제가 생긴다. 자, 한번 살펴보자.

**Main.storyboard**를 열고, **File Inspector**에서 Auto Layout과 Size Classes를 끄자.  
![uncheck autolayout & sizeclass](http://cdn1.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate4-202x320.png)

**Use Auto Layout** 의 체크를 풀면 사이즈 클래스도 못쓴다는 경고가 나온다. 괜찮다. 자, 이제 스토리보드는 구식의 struts-and-springs 모델을 사용한다.

> **Note: **오토레이아웃은 iOS 6에서 등장했다. iOS 5에서 작동하게 하려면 오토레이아웃을 풀어주자. 사이즈 클래스는 iOS 7부터다.

메인 뷰에 새로운 뷰들을 구성하자:  
![views](http://cdn4.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate5-382x500.png)

  * **Top left:** Origin 20,20, Size 130 x 254
  * **Top right:** Origin 170,20, Size 130 x 254
  * **Bottom:** Origin 20,294, Size 280 x 254

앱을 실행시키지 않고서도 **Preview Assistant**를 통해 실행되었을때의 모습을 확인할 수 있다. **Assistant editor**를 열고 어시스턴트 에디터의 탑바에서 버튼을 눌러 **Preview/Main Storyboard**로 바꿔주자(기본적으로 Automatic으로 되어 있을 것이다).

![preview assistant](http://cdn1.raywenderlich.com/wp-content/uploads/2014/09/OpeningPreview.png)

> 내가 현재 사용하고 있는 Xcode 6.1.1에서는 아이콘이 다르다. 어시스턴트 에디터는 **View &gt; Show Assistant Editor** 로 열 수 있다.

좌하단의 **+** 버튼을 통해 다양한 스크린 사이즈를 추가할 수 있고, 각 스크린을 클릭하면 뷰를 회전시킬 수도 있다! 두개의 4인치 스크린을 세팅하고, 하나는 landscape로 눕히자.

![landscape](http://cdn3.raywenderlich.com/wp-content/uploads/2014/09/LandscapeWrong.png)

이걸

![landscape_good](http://cdn4.raywenderlich.com/wp-content/uploads/2013/09/StrutsProblem-landscape-looks-good-480x289.png)

이렇게 보이게 하고 싶다.

지금의 문제는 세 뷰에 전부 오토사이징 마스크를 걸었다는 점이다. 오토사이징 설정을 바꿔주자:  
![top-left](http://cdn1.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate7-250x250.png) ![top-right](http://cdn1.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate8-250x250.png) ![bottom](http://cdn4.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate9-250x250.png)

좌표를 보면 알겠지만, 왼쪽부터 차례대로 top-left, top-right, bottom이다. 다 하고 나면 이렇게 바뀐다:

![result](http://cdn4.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdat10-480x320.png)

좀 나아졌지만 여전히 부족하다. 뷰간의 패딩이 맞지 않는다. 오토사이징 마스크는 수퍼뷰가 리사이즈될때 자식뷰에게 리사이즈하라고 알려주지만, 어떻게 리사이즈해야 할지에 대해서는 전혀 알려주지 않는다.

flexible width, height 세팅들(the "springs")을 건드려볼 수는 있겠지만, 어떻게 하든 정확히 20 포인트의 뷰간 간격은 맞출 수 없을 것이다.

![why](http://cdn1.raywenderlich.com/wp-content/uploads/2012/09/Rage-why.png)

springs and struts method로 이러한 문제를 해결하기 위해서는 어쩔 수 없이 코드를 작성해야 한다. UI가 변하면, **viewWillLayoutSubviews** 함수가 메시지를 받는다. 이 함수를 오버라이드하자. 코드를 작성하기 전에, outlet property를 통해 스토리보드와 코드를 연결해주자.

어시스턴트 에디터 모드를 변경하고 **ViewController.swift**를 열어서 outlet을 연결해주자.

![outlet](http://cdn4.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate11-700x229.png)
    
    
    @IBOutlet weak var topLeftView: UIView!
    @IBOutlet weak var topRightView: UIView!
    @IBOutlet weak var bottomView: UIView!
    

다 되면 아래 코드도 추가해주자:
    
    
    override func viewWillLayoutSubviews() {
    
      if UIInterfaceOrientationIsLandscape(self.interfaceOrientation) {
        var rect = self.topLeftView.frame
        rect.size.width = 254
        rect.size.height = 130
        self.topLeftView.frame = rect
    
        rect = self.topRightView.frame
        rect.origin.x = 294
        rect.size.width = 254
        rect.size.height = 130
        self.topRightView.frame = rect
    
        rect = self.bottomView.frame
        rect.origin.y = 170
        rect.size.width = 528
        rect.size.height = 130
        self.bottomView.frame = rect
      }
      else {
        var rect = self.topLeftView.frame
        rect.size.width = 130
        rect.size.height = 254
        self.topLeftView.frame = rect
    
        rect = self.topRightView.frame
        rect.origin.x = 170
        rect.size.width = 130
        rect.size.height = 254
        self.topRightView.frame = rect
    
        rect = self.bottomView.frame
        rect.origin.y = 295
        rect.size.width = 280
        rect.size.height = 254
        self.bottomView.frame = rect
      }
    }
    

이렇게 주구장창 코딩을 하고 나면 이제 잘 작동하는 것을 볼 수 있다.

> 원문에서는 이것만 하고 실행하면 크래시가 나고, 오토사이징 부분을 바꿔줘야 된다는데 나는 그냥 해도 잘 된다. 만약 크래시가 나면 원문을 참조하자.

그런데 이는 아직 4인치 스크린 사이즈에 대해 코딩했을 뿐이다. 각 스크린 사이즈에 맞춰 다 코딩해줘야 한다. 그게 싫다면, portrait와 landscape 각각 하나씩 두개의 xib를 만드는 방법도 있다. 어떤 방법을 선택하든 골치아프다.

## Auto Layout to the rescue!

자 그럼 오토레이아웃으로 구현해보자. 위에서 작성한 **viewWillLayoutSubviews** 부분을 지우자. 코딩 없이 구현할 것이다!

**Main.storyboard** 에 가서 **Use Auto Layout**과 **Use Size Classes** 옵션을 켜주자:

![enable options](http://cdn1.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate13-210x320.png)

### A quick note on Size Classes

사이즈 클래스는 iOS 8과 Xcode 6에서 새로 등장했다. 이것을 이용하면 유니버셜 앱을 하나의 스토리보드로 직관적으로 구현할 수 있다. 스크린에서 볼 수 있는 거의 모든 것들이 사이즈클래스를 가질 수 있다 - 스크린, 뷰, 뷰컨트롤러를 포함해서. 사이즈클래스에는 두 타입이 있다: vertical과 horizontal. 각 vertical과 horizontal은 **Regular**, **Compact**, **Any** 셋중 하나의 값을 가진다.

사이즈 클래스는 우리의 앱이 작동하는 디바이스와 orientation에 따라 달라진다. 예를 들어, portrait 아이폰은 **Regular** height와 **Compact** width를 가진다. Any값은 제네릭한 사이즈 클래스의 값으로 사용된다. 즉 모든 레이아웃의 수퍼클래스라고 생각하자. 만약 현재 디바이스와 orientation에 대해 지정된 사이즈 클래스가 없다면, 스토리보드는 Any로부터 레이아웃을 끌어온다.

스토리보드의 중앙 하단에 "w**Any** h**Any**"가 현재 사이즈 클래스 설정 상태를 보여준다. 클릭하면 사이즈클래스 설정 그리드를 볼 수 있다:  
![sizeclass config gird](http://cdn2.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate2-458x320.png)

애플은 유니버셜 앱을 만들 때 초기 UI 레이아웃을 이 제네릭 사이즈 클래스, 즉 Any 상태에서 구성하길 추천한다. 모든 사이즈 클래스가 여기서 시작하기 때문에. 우리의 스토리보드 설정이 "w**Any** h**Any**"인지 확인하자.

우리의 제네릭 사이즈 클래스 설정에 따라 정사각형으로 변하는 스토리보드의 씬의 사이즈를 주목하자.

사이즈클래스에 대해 더 자세히 알고 싶으면 [Beginning Adaptive Layout Tutorial](http://www.raywenderlich.com/83276/beginning-adaptive-layout-tutorial)를 참고하자. 뭐 이리 볼게 많나 싶지만 아무튼 이 포스트는 오토레이아웃의 기초에 대한 포스트다.

### Your First Auto Layout Constraints

preview assistant에서 landscape layout을 보자. 이렇게 보일 것이다:  
![landscape](http://cdn3.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate14-480x270.png)

자 그럼 오토레이아웃을 시도해보자. 아래 스샷처럼 위 두 뷰를 선택하고 **Editor/Pin/Widths Equally**를 선택하자:  
![widths equally](http://cdn1.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate15-631x500.png)

마찬가지로 **Editor/Pin/Horizontal Spacing**도 설정해주자:  
![horizontal spacing](http://cdn2.raywenderlich.com/wp-content/uploads/2014/09/StrutsUpdate1-530x500.png)

주황색 "T-bar"모양의 선은 **constraints**를 나타낸다. 지금까지 우리는 Equal Widths와 Horizontal Space 두가지의 constraints를 걸었다. constraints는 뷰간의 관계를 나타내고, 오토레이아웃을 사용해서 레이아웃을 만드는 첫번째 툴이다. 좀 어려워 보일 수 있지만 의미를 이해하고 나면 굉장히 직관적임을 깨달을 것이다.

그럼 이제 아래 스텝들을 따라 레이아웃을 완성해보자. 각 스텝마다 주황색 T-bar가 추가될것이다. 매번 뷰들을 재선택해줘야 한다는 걸 잊지 말자.

Top-left의 초록색 뷰에 대해, **Editor/Pin**에서:

  * Top Space to Superview
  * Leading Space to Superview

Top-right의 노란색 뷰:

  * Top Space to Superview
  * Trailing Space to Superview

Bottom의 파란색 뷰:

  * Leading Space to Superview
  * Trailing Space to Superview
  * Bottom Space to Superview

다 하고 나면 이렇게 된다:  
![constrainted](http://cdn1.raywenderlich.com/wp-content/uploads/2014/09/StrutsUpdate2-538x500.png)

constraints를 나타내는 T-bar가 파랗게 변했지만, 몇개는 여전히 주황색으로 남아있다. 이는 레이아웃이 아직 미완성이라는 것을 의미한다. 미완성이란, 오토레이아웃이 뷰의 포지션과 사이즈를 계산할 수 있을 만큼 충분히 constraints가 제공되지 않았다는 의미다. 다 파랗게 될때까지 constraints들을 추가해주자.

뷰 세개를 다 선택하고, **Pin/Heights Equally**를 걸어주자. 그리고 top-left뷰와 bottom뷰를 선택하고 **Editor/Pin/Vertical Spacing**을 걸어주자. 이제 주황색이 사라졌다:

![blues](http://cdn5.raywenderlich.com/wp-content/uploads/2014/09/StrutsUpdate3-570x500.png)

이는 오토레이아웃이 정상적인 레이아웃을 구성할 수 있을 만큼 충분한 정보를 제공받았다는 것을 의미한다. 그러나 아직도 만족스럽지 않다 - 스크린의 오른쪽에 제네릭 사이즈 클래스로 변경하면서 발생한 커다란 스페이스가 남아있다. bottom 뷰의 trailing space constraint를 선택하자:

![trailing space cosntraint](http://cdn5.raywenderlich.com/wp-content/uploads/2014/09/Strutsupdate4.png)

사이즈 인스펙터를 열고 **Constant** 값을 **20**으로 바꿔주자.

![size inspector](http://cdn2.raywenderlich.com/wp-content/uploads/2014/09/StrutsUpdate5.png)

Top-right 뷰에 대해서도 똑같이 해 주고 나서 프리뷰 레이아웃을 보면, 완벽하다! portrait, landscape에서 모두 훌륭하게 작동할 뿐만 아니라 디바이스의 사이즈에 상관없이 퍼펙트하게 작동한다. 심지어 아이패드에서도 문제없다!

![iPad](http://cdn4.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayout19-480x270.png)

좋아, 그럼 우리가 뭘 했는지 되짚어보자. 오토레이아웃은 뷰의 포지션과 사이즈를 하드코딩 하지 않고, 뷰간의 그리고 수퍼뷰와의 constraints를 사용한다. 이 constraints는 결국 뷰와 뷰 사이의 상대적인 포지션과 사이즈를 의미하며 우리가 지금까지 적용한 constraints들은 다음과 같다:

  * top-left, top-right 뷰는 항상 같은 width를 가진다(pin widths equally).
  * top-left와 top-right 뷰 사이에는 20포인트의 수평 패딩이 있다(pin horizontal spacing).
  * 모든 뷰는 같은 height를 가진다(pin heights equally).
  * 위의 두 뷰와 아래 바텀뷰 사이에는 20포인트의 수직 패딩이 있다(pin vertical spacing).
  * 각 뷰들과 스크린 엣지 사이에는 20포인트의 마진이 있다(top, bottom, leading &amp; trailing space to superview constraints).

스토리보드 좌측의 Document Outline에서 모든 constraints를 볼 수 있다. 오토레이아웃을 키면 Xcode가 여기에 **Constraints**라는 섹션을 추가한다. 만약 outline창이 보이지 않는다면 인터페이스 빌더 윈도우 좌측 하단의 버튼을 클릭하자.

Document Outline에서 constraint를 클릭하면 인터페이스 빌더에서 해당 constraint를 하이라이팅 해준다:

![constraints highlight](http://cdn5.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate20-480x281.png)

constraints는 **NSLayoutConstraint**라는 클래스의 오브젝트이고 따라서 attributes를 가지고 있다. 예를 들어 두 top 뷰 사이의 패딩 constraint(Horizontal Space 20 중 하나)를 선택하고 **Attributes inspector**로 가면 **Constant** 필드를 수정하여 마진의 사이즈를 변경할 수 있다.

![attributes inspector](http://cdn1.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate21-358x320.png)

이렇게 수정하면 넓어지는 걸 확인할 수 있다:

![wider](http://cdn5.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate22-480x270.png)

오토레이아웃은 뷰들을 구성하는 데 있어서 springs and struts 모델보다 훨씬 강력하다. 이제 본 튜토리얼의 나머지 부분에서는 constraints의 모든 것에 대해서 그리고 이를 이용해서 어떻게 다양한 레이아웃을 구성하는 지 배울 것이다.

## How Auto Layout Works

처음에 오토사이징 마스크를 이용해서 만들었을 때는 이렇게 만들었다:

![position&size](http://cdn4.raywenderlich.com/wp-content/uploads/2013/09/StrutsProblem-coordinates.png) ![autosizing mask](http://cdn4.raywenderlich.com/wp-content/uploads/2012/09/Struts-coordinates-2-353x500.png)

오토레이아웃에서는, 이렇게 한다:

![autolayout](http://cdn1.raywenderlich.com/wp-content/uploads/2012/09/Struts-coordinates-3-353x500.png)

뷰의 사이즈와 포지션은 더이상 중요하지 않다. 물론 캔버스에 새로운 버튼이나 라벨을 넣을 때 특정한 사이즈를 가지고 특정한 위치에 놓긴 하겠지만, 이는 인터페이스 빌더에게 constraints를 넣을 곳을 알려주기 위한 도구에 불과하다.

오토레이아웃에 들어 있는 아이디어는 어디에 어떤 수치를 지정할것인지 그리고 상대적으로 나머지 레이아웃들을 배치하는 것을 단순하게 만들어준다.

## Designing by Intent

constraints를 사용하는 큰 장점 중 하나는, 뷰들이 적당한 곳에 위치하도록 삽질하며 시간을 버릴 필요가 없다는 것이다. 대신 오토레이아웃에게 뷰간의 관계를 알려주면 오토레이아웃이 그 일을 다 해준다. 이를 **designing by intent**라 한다.

이를 한국말로 번역하자면 "의도에 따른 디자인"이라고 할 수 있겠다. 말 그대로 의도대로, 직관적으로 디자인 할 수 있다는 얘기다. "design by intent"을 하면, 우리는 디자인을 어떻게 해야 하는지를 표현할 필요 없이 어떻게 되었으면 하는지만 표현하면 된다. 무슨 말이냐면, "버튼을 왼쪽 위 코너로부터 (20, 230) 되는 위치에 놓아 주세요" 라고 말하는 대신 "버튼을 수퍼뷰의 수직으로 중앙에, 그리고 수퍼뷰의 왼쪽 엣지로부터 20픽셀 떨어진 곳에 놓아 주세요" 라고 말하면 된다는 거다. 그러면 오토레이아웃은 수퍼뷰의 크기에 상관없이 알아서 계산해서 배치해 준다. 즉, 이전에 비해 더욱 서술적으로 UI를 디자인할 수 있다. 

또다른 오토레이아웃의 장점은 국제화다. 언어마다 길이가 달라서 골치가 아픈데, 오토레이아웃이 constraints에 맞춰 텍스트의 길이를 고려해서 라벨의 사이즈를 조정해준다. iOS 7에서 유저가 글로벌 텍스트 사이즈를 조정할 수 있게 됨에 따라 이러한 중요성은 더욱 증가했다.

자, 그럼 오토레이아웃에 더 익숙해져 보자.

## Courting constraints

Single View Application 템플릿으로 새 프로젝트를 만들자. 이름은 "Constraints"라고 하자.

Xcode 6은 디폴트로 오토레이아웃을 사용한다. 일단 **Main.storyboard**를 열고 사이즈 클래스를 끄자(disable). 

> 사이즈 클래스는 왜 끄지?

캔버스에 새로운 버튼을 올려놓자. 드래그를 해 보면 점선 **guides**가 나타날 것이다.  
![guides](http://cdn2.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate23-501x500.png) ![guides2](http://cdn5.raywenderlich.com/wp-content/uploads/2013/09/Constraints-other-guides.png)

자, 그럼 이제 새로운 오브젝트가 생겼다. 새로 만든 오브젝트는 아무 constraints를 가지고 있지 않다. 우리가 배운 바에 따르면 오토레이아웃은 항상 사이즈와 포지션을 결정할 수 있는 충분한 constraints를 요구한다. 그러나 이것은 작동한다! 분명히 미완성 레이아웃일텐데?

이는 디폴트 constraints로 **automatic constraints**라는 것이 할당되기 때문이다. 이는 디자인 타임은 아니고 컴파일 타임에 수행된다1. 오토매틱 cosntraints는 오브젝트의 사이즈와 포지션을 고정된 위치에 할당한다. 오토레이아웃을 무시하고 고정된 사이즈와 포지션을 사용할 일이 꽤 많기 때문에, 이는 상당히 유용하다.

자, 그럼 이제 constraints를 가지고 놀아보자. 현재 버튼은 좌상단에 no constraints 상태다. 가이드에 맞춰져 있는지 다시한번 확인하자. **Editor/Pin**에 가서 아래와 같이 되도록 두 constraints를 할당하자:  
![constraints](http://cdn2.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate25-480x221.png)

알겠지만, **Leading Space to Superview** 와 **Top Space to Superview** 이다. 그리고 이 constraints는 아까 봤던 것처럼 Document Outline에서 확인할 수 있다:

![document outline](http://cdn2.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate26-475x320.png)

자 그럼 버튼을 우상단으로 옮겨보자:  
![top-right](http://cdn3.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate27-480x215.png)

T-bar가 angry orange로 변했다! 이는 IB에 있는 버튼의 사이즈 및 포지션이 constraints에 기반해서 오토레이아웃이 기대하는 사이즈 및 포지션과 다르기 때문에 발생하는 문제다2. 이를 **misplaced view**라 한다.

앱을 실행시켜 보면 버튼은 여전히 좌상단 코너에 있는 걸 확인할 수 있다:

![misplaced view](http://cdn4.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate28-480x192.png)

오토레이아웃에서, _orange는 나쁘다_. 위에서 IB가 두개의 오렌지 박스를 그렸다: 하나는 점선, 하나는 직선. 점선은 오토레이아웃에 의한 뷰의 프레임을 의미한다. 실선은 씬(IB)에서 우리가 갖다 놓은 뷰의 프레임을 의미한다.

Horizontal Space constraint를 지우자. 캔버스(IB)에서나 도큐먼트 아웃라인에서 선택하고 **Delete**키를 눌러 지울 수 있다.

![delete horizontal space cosntraint](http://cdn5.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate29-480x217.png)

버튼에 오렌지 아웃라인이 잡힌 것과 점선 레드 박스를 주목해라. 오렌지 아웃라인은 뭔가 잘못되었다는 것을 의미하고, 점선 레드 박스는 오토레이아웃이 생각하기에 런타임에 버튼이 위치할 곳을 나타낸다. 이는 constraints가 부족해서 발생하는 문제다.

> "점선 레드 박스" 가 뭔지 모르겠는데, 그냥 Button 밑의 빨간 밑줄을 의미하는 거 같다.

X 포지션에 대해 Xcode가 오토매틱 constraints를 할당할 줄 알았는데, 왜 그렇지 않을까? 오토매틱 cosntraints는 우리가 직접 constraints를 전혀 할당하지 않았을 때만 부여된다. 우리가 constraints를 하나라도 할당하면 그때부터는 완전히 제어권한이 우리에게 넘어오게 된다.

버튼을 선택하고 **Editor\Pin\Trailing Space to Superview**를 넣어주자. 알겠지만, 우측으로부터 일정 거리를 fix하겠다는 의미다. 앱을 실행시키고 디바이스를 회전시켜 보면 잘 작동하는 것을 확인할 수 있다.

자 그럼 이번엔 버튼을 왼쪽으로 좀 밀어 보자:  
![left](http://cdn4.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate31-480x200.png)

마찬가지로, misplaced view 때문에 점선 오렌지 박스가 등장했다. 새로운 위치로 옮기고 싶다면 어떻게 해야 할까? cosntraint를 지운 다음 다시 만드는 것도 하나의 방법일테지만 더 쉬운 방법이 있다: **Editor/Resolve Auto Layout Issues/Update Constraints**.  
![updated](http://cdn3.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate32-480x255.png)

좌측 Document Outline에서도 수치가 업데이트된 것을 확인할 수 있다:  
![updated2](http://cdn5.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate33-462x320.png)

지금까지 Horizontal Space와 Vertical Space constraints만 해왔다. 이제 "Center" constraint를 해 보자. 기존의 constraints는 지우고, 버튼을 가이드(IB guides)에 따라 하단 중앙으로 옮기자. 그리고 **Editor/Align/Horizontal Center in Container**로 수퍼뷰의 가운데로 정렬할 수 있다. 아래처럼 오렌지 라인이 나타난다:

![orange line](http://cdn1.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate35-480x237.png)

X축에 대해서 위치는 정해줬지만 Y축에 대해서는 위치가 정해지지 않았으므로 오렌지 라인이다. Y축에 대해서도 정해주자:

![bottom](http://cdn4.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate36-480x227.png)

이제 실행시켜서 회전시켜 보면 잘 동작하는 걸 볼 수 있다.

오토레이아웃은, 지금까지 말해왔듯이, 사이즈와 포지션을 지정해주지 않아도 constraints만 정해주면 이에 맞춰 뷰를 구성한다. 이러한 패러다임의 변화를 버튼의 **Size inspector**에서 볼 수 있다:  
![paradigm change](http://cdn1.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate38-563x500.png)

이번엔 버튼의 width를 바꿔보자. 우측 inspector에서 width를 그냥 바꾸면 view misplaced가 발생한다. width를 바꿀 때도 constraint를 바꿔줘야 한다. **Editor/Pin/Width**를 눌러서 바꿔주면 잘 되는 것을 볼 수 있다. 이렇게 한번 width constraint를 걸어주면, 그 이후로는 이전과 같이 버튼 아래의 T-bar를 선택하거나 좌측의 Document Outline에서도 변경할 수 있다:

![document outline](http://cdn5.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate42-269x320.png)

그러면 우리가 cosntraint를 지정해주기 이전 버튼의 width는 어떻게 정해진걸까? constraint가 없다면 버튼은 내부 텍스트의 길이에 따라 자동으로 width를 정한다. 버튼의 background image를 설정했다면 거기에 맞춘다.

이를 **intrinsic content size**라 한다. 모든 컨트롤들이 이를 지원하는건 아니지만 UILabel을 포함해서 많은 것들이 그렇다. 버튼을 원래대로, 즉 intrinsic content size로 돌려놓기 위해서 지정한 width constraint를 지우고 **Editor/Size to Fit Content**를 클릭하자.

## It takes two to tango (손바닥도 마주쳐야 소리가 난다)

가이드는 자식뷰와 수퍼뷰 사이에만 나타나는 것이 아니라 같은 레벨의 뷰간에서도 나타난다. 버튼을 하나 더 올려보자:

![buttons](http://cdn1.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate43-480x185.png)

이제 버튼간의 constraint를 줘야 하는데, **Editor/Pin**에서 줄 수도 있지만 더 쉬운 방법이 있다. 버튼을 선택하고 **Ctrl-drag**를 해서 다른 버튼에 놓자:

![ctrl-drag](http://cdn3.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate44-480x214.png)

그럼 아래와 같은 팝업에서 **Horizontal Spacing**을 선택하자:

![popup](http://cdn3.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate45-325x320.png)

이제 constraint가 생겼지만 여전히 오렌지다. 사이즈는 자동으로 정해지고, x좌표는 방금 정해 주었는데, y좌표가 아직 미정이다. Document Outline의 우상단 빨간색 화살표를 누르면 이를 쉽게 확인할 수 있다:

![layout issues](http://cdn2.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate47-480x257.png)

좋아, 이제 y좌표 constraint를 걸어주자. 새로운 버튼에서 **Ctrl-drag**를 해서 아래쪽 뷰에 놓아주자:

![ctrl-drag](http://cdn2.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate48-480x162.png)

아까랑 조금 다른 팝업 옵션이 보인다. **Bottom Space to Bottom Layout Guide**를 선택해주자. 바텀으로부터 Vertical Space가 잡힐 것이다. 이제 여기서 첫번째 버튼(가운데 있는 버튼)의 텍스트를 늘려보자. 버튼을 더블클릭하고 내용을 A longer label로 수정하면 우리가 지정했던 constraint에 따라 변하는 모습을 볼 수 있다3:

![longer label](http://cdn4.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate50-700x234.png)

이 파트의 마지막으로, 버튼간 Left edge를 맞추는 방법을 소개한다. 일단 버튼을 하나 더 만들고, 헷갈리지 않도록 색을 입힌다:

![buttons](http://cdn2.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate51-700x258.png)

초록색 버튼에서 **Ctrl-drag**를 해서 노란색 버튼에 놓고, **Vertical Spacing**을 켜주자. 그렇게 만들어진 Vertical Space Constraint는 **Attributes inspector**에서 constraint 크기를 수정할 수 있다. 이걸 40으로 늘려보면 아직 x좌표 constraint가 없기 때문에 IB에서는 오렌지 박스가 나타나고 변화가 안보이지만, 실행시켜 보면 적용이 된 걸 확인할 수 있다.

> 원문 튜토리얼에서는 IB에서도 적용이 되는 거 같음.

![run](http://cdn5.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate53-480x270.png)

근데 보다시피, x좌표 constraint를 안걸어줬기 때문에 이상하게 나온다. 이제 이걸 고쳐보자. 아래 노란 버튼과 같은 x좌표를 가지면 예쁠 것 같다. 일단 IB에서 위치를 맞춰주고, **Ctrl-drag**를 해서 **Left**를 선택하자. 이 constraint는 두 오브젝트의 왼쪽 엣지가 항상 같다 라는 의미다.

![final](http://cdn1.raywenderlich.com/wp-content/uploads/2014/09/SwiftAutoLayoutUpdate56-480x205.png)

## Next?

와, 드디어 다 끝났다! 설명이 상세해서 좋긴 한데, 좀 지나치게 상세하다. 어차피 튜토리얼일 뿐인데. [파트 2](http://www.raywenderlich.com/?p=83130)는 번역을 안할 것 같다. 아니면 정말 간단하게 요약해서 정리하던지…

당연한 얘기지만, 튜토리얼도 좋지만 이쯤 했으면 그냥 부딪히면서 만드는게 좋다!

* * *

  1. 디폴트로 차 있는 건 아니고, 컴파일 타임에 constraints가 없으면 부여된다는 소린 거 같다.↩

  2. 즉, IB에서 함부로 옮기면 곤란하다는 뜻이다. constraints가 가해진 오브젝트는 constraints를 통해 수정하자.↩

  3. Attribute Inspector에서도 내용을 수정할 수 있는데, 여기서 수정하면 intrinsic content size가 발동하지 않는다. ↩


[Tistory 원문보기](http://khanrc.tistory.com/82)
