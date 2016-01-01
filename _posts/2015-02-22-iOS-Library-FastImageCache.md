---
layout: post
title: "iOS Library: FastImageCache"
tags: ['iOS']
date: 2015-02-22 20:10:00
---
# iOS Library: [FastImageCache](https://github.com/path/FastImageCache)

패스트 이미지 캐쉬는 [Path](http://www.path.com/)에서 공개한 라이브러리로써 빠른 이미지 로드를 돕는다. 이미지를 Uncompressed 상태로 저장하기 때문에 작은 크기의 이미지에 효율적이다.

자세한 사용법은 위 링크를 참조하자. 여기서는 개념만 소개한다. 위 링크 일부의 번역에 해당.

**참고**: 벤치마크 결과  
[iOS image caching. Libraries benchmark (SDWebImage vs FastImageCache)](https://bpoplauschi.wordpress.com/2014/03/21/ios-image-caching-sdwebimage-vs-fastimage/)  
disk에서 이미지를 불러올 때 탁월하게 빠른 성능을 보여준다.

## 패스트 이미지 캐쉬는 뭘 할까?

  * 비슷한 사이즈/스타일의 이미지들을 함께 저장
  * 디스크에 데이터 보존
  * 기존 방법보다 빠르게 이미지를 반환
  * 이미지의 사용 시기에 따라 자동 캐쉬 삭제 (쓴지 오래 된 건 자동으로 지움)
  * 이미지의 저장 및 로드에 대해 Model-based approach
  * 이미지가 캐쉬에 저장되기 전에 모델별 프로세싱 가능 (이해 안 감)

## 패스트 이미지 캐쉬는 어떻게 작동할까?

패스트 이미지 캐쉬를 이해하기 위해, 기존의 이미지를 다루는 방법들이 어떠한가를 먼저 알 필요가 있다.

### The Scenario

API를 통해 이미지를 불러오고, 해당 이미지의 사이즈와 스타일을 원하는 형태로 처리한 후 디바이스에 저장한다. 이후에 어플리케이션이 이 이미지가 필요하면, 디스크로부터 메모리로 이미지를 로드하고 이미지뷰 등을 통해 화면에 렌더링한다.

### The Problem

자 그럼 여기서 문제가 발생한다. 저장되어 있는, 즉 압축된 디스크상의 이미지 데이터를 유저가 볼 수 있도록 코어 애니메이션 레이어로 렌더링하는 작업이 매우 비싸다. 여기에 스크롤뷰까지 추가되면 컨텐츠(이미지)들이 빠르게 변하기 때문에 문제는 더 심각해진다. 아무튼, 부드러움을 유지하려면 60FPS를 유지해야 하기 때문에. 

디스크로부터 이미지를 로드해서 스크린에 보여주는 과정은 다음과 같다:

  1. `+[UIImage imageWithContentsOfFile:]` 이 [Image I/O](https://developer.apple.com/library/ios/documentation/graphicsimaging/conceptual/ImageIOGuide/imageio_intro/ikpg_intro.html#//apple_ref/doc/uid/TP40005462-CH201-TPXREF101) 를 사용해서 memory-mapped data로부터 `CGimageRef`를 만든다. 이 시점에서, 이미지는 아직 decoded되지 않았다.
  2. 반환받은 이미지를 `UIImageView`에 할당한다.
  3. An implicit `CATransaction` captures these layer tree modifications. (이해 안 감. 아무튼 내포된 CATransaction이 무언가 처리를 한다는 듯)
  4. 이후에 Core Animation이 내포된 트랜잭션을 처리한다. 이 트랜잭션은 레이어에 셋팅된 이미지의 카피를 만드는 것을 포함한다. 이미지를 카피하는 것은 아래 스텝의 전부 또는 일부로 구성된다:  
i. file IO와 decompression을 위한 버퍼가 할당된다.  
ii. 디스크로부터 파일을 읽어 메모리에 로드한다.  
iii. 압축된 이미지를 무압축 비트맵 형태로 decode한다. 이는 매우 CPU 바운드 작업이다.  
iv. 코어 애니메이션이 무압축 비트맵 데이터를 사용해서 레이어에 렌더링한다.

이는 매우 느릴 수 있다!

### The Solution

패스트 이미지 캐쉬는 위 과정중 많은 부분을 생략할 수 있다. 다양한 테크닉을 사용해서:

#### Mapped Memory

패스트 이미지 캐쉬의 핵심은 이미지 테이블이다. 이미지 테이블은 2D 게임에서 사용되는 [sprite sheets](http://en.wikipedia.org/wiki/Sprite_\(computer_graphics\)#Sprites_by_CSS)와 같다. 이미지 테이블은 같은 차원의 이미지들을 한 파일에 넣는다. 이 파일은 어플리케이션이 시작해서 종료될때까지 I/O를 위해 항상 열려있게 된다.

이미지 테이블은 [mmap](https://developer.apple.com/library/ios/documentation/System/Conceptual/ManPages_iPhoneOS/man2/mmap.2.html) 시스템 콜(system call)을 사용한다. 이 시스템 콜은 파일 데이터를 메모리로 다이렉트로 매핑한다. 이 시스템 콜은 memcpy와는 다르게 단순히 디스크상의 데이터와 메모리 공간을 매핑한다.

이미지 캐쉬로 리퀘스트가 들어오면, 이미지 테이블은 상수시간에 해당 이미지의 위치를 찾아 메모리로 매핑한다. 그러면 backing store로 매핑된 파일 데이터를 갖는 `CGImageRef`가 만들어진다.

이러한 형태를 mapped memory라고 하는데, 이 mapped memory는 다양한 장점을 갖고 있다. 먼저 iOS의 버추얼 메모리 시스템이 파일 단위로 페이징하기 때문에(의역, 오역 가능성 있음), VM 시스템이 알아서 메모리를 관리해준다. 또한 mapped memory는 어플리케이션이 실제로 사용하는 메모리에 포함되지 않는다.

비슷한 방식으로, 이미지 데이터가 이미지 테이블에 저장되면 memory-mapped bitmap context가 만들어진다. 오리지날 이미지와 이 컨텍스트는 함께 하나의 엔티티 오브젝트로 이미지 테이블로 전달된다. 이 오브젝트는 컨텍스트에 바로 뿌릴 수도 있고, 편집도 가능하다.

#### Uncompressed Image Data

decompression 작업은 비싸기 때문에, 이미지 테이블은 무압축 이미지 데이터를 파일에 저장한다. 이는 퍼포먼스 향상 뿐만 아니라 [utilize image format families](https://github.com/path/FastImageCache#working-with-image-format-families)도 가능케 한다.

이 방식의 문제 또한 명확한데, 당연하게도 디스크 용량을 더 많이 잡아먹는다. 특히 JPEG같은 포멧에 대해서는 더더욱. 이런 이유로 **패스트 이미지 캐시는 작은 크기의 이미지에 대해서 좋은 성능을 보인다**.

#### Byte Alignment

고성능 스크롤링을 위해서, 코어 애니메이션이 이미지를 카피하지 않고 직접 사용할 수 있도록 하는 것은 중요하다. 코어 애니메이션이 이미지를 카피하는 이유 중 하나는 `CGImageRef`의 부적합한 byte-alignment에 있다. 이미지 테이블은 각각의 이미지에 맞게 적합한 byte-alignment를 설정한다. 결과적으로 이미지를 로드하면 코어 애니메이션이 추가작업 없이 바로 쓸 수 있다!

## 고려사항들

### Image Table Size

이미지 테이블이 가질 수 있는 최대 이미지 개수는 이미지 포멧1에 의해 결정된다. 이미지 테이블은 픽셀당 4바이트를 할당하므로 이미지 테이블 파일이 차지하는 최대 용량은 다음과 같다:

`픽셀당 4바이트 * 이미지의 픽셀 수 * 이미지 테이블의 최대 이미지 개수`

패스트 이미지 캐쉬를 사용하는 앱은 각 이미지 테이블이 몇개의 이미지를 가질 것인가를 신중히 고려해야 한다. 이미 꽉 찬 이미지 테이블에 새로운 이미지를 저장하려 한다면, least-recently-accessed2 이미지에 덮어씌운다. 

### Image Table Transience(일시적임)

이미지 테이블 파일은 유저의 caches 디렉토리 안의 `ImageTables`라는 서브디렉토리에 저장된다. iOS는 디스크 공간을 확보하기 위해 캐쉬 파일을 언제든 지울 수 있고, 따라서 패스트 이미지 캐쉬를 사용하는 앱은 이 점을 염두에 두고 이미지 테이블 파일이 사라졌을 때를 대비해야 한다.

> **Note**: 유저의 caches 디렉토리 안의 파일들은 iTunes나 iCloud로 백업되지 않는다

### Source Image Persistence

패스트 이미지 캐쉬는 오리지널 이미지를 이미지 데이터로 가공하여 이미지 테이블에 저장한다. 즉, 오리지널 이미지는 따로 보존하지 않는다.

예를 들어, 오리지널 이미지로 섬네일을 만들고 이를 이미지 테이블에 저장한다고 하자. 이 때 오리지널 이미지를 후에 다시 사용할 수 있도록 보존할 책임은 어플리케이션에게, 즉 우리에게 있다.

싱글 소스 이미지를 효율적으로 사용하기 위해서 이미지 포멧 패밀리가 명시될 수 있다. [Working with Image Format Families](https://github.com/path/FastImageCache#working-with-image-format-families)를 참고하자. 이미지 포멧을 명시함으로써 더 효율적인 이미지 관리가 가능하다는 의미인 것 같다.

### Data Protection

iOS 4에서, data protection이 등장했다. 디바이스가 잠기거나 꺼지면, 디스크가 암호화된다. 그런데 iOS 7에서 백그라운드 모드가 등장했고, 디스크가 암호화된 상태에서 앱이 파일에 어세스하려고 하는 이슈가 발생한다.

패스트 이미지 캐쉬는 이미지 테이블 파일을 만들 때 각 이미지 포멧에 데이터 프로텍션 모드를 지정할 수 있다. 이미지 테이블 파일의 데이터 프로텍션을 키면 디스크가 암호화 되어 있을 때 이미지 데이터를 읽고 쓰지 못하게 된다 (이해 잘 안감).

## Requirements

  * iOS 6.0 ~
  * use ARC
  * Demo requires Xcode 5.0 ~

## Getting Started

일단 코코아팟으로 깔자. 직접 받아서 추가하는것도 썩 복잡하지 않은 듯.

### Initial Configuration

앱의 실행마다 이미지 캐쉬를 사용하기 전에 설정이 필요하다. 앱델리게이트에 적당히 배치하자.

#### Creating Image Formats

각 이미지 포멧은 이미지 캐쉬에서 사용할 이미지 테이블과 일치한다(?). 이미지 테이블에 저장하는, 즉 화면에 렌더링하는 이미지를 만들기 위해 같은 오리지널 소스 이미지를 사용하는 이미지 포멧은 같은 이미지 포멧 패밀리에 속한다. 즉, 오리지널 이미지가 하나 있고 이걸 리사이징 해서 미디움 사이즈의 썸네일과 스몰 사이즈의 썸네일을 만든다면 이 두 썸네일은 같은 이미지 포멧 패밀리에 해당한다.
    
    
    static NSString *XXImageFormatNameUserThumbnailSmall = @"com.mycompany.myapp.XXImageFormatNameUserThumbnailSmall";
    static NSString *XXImageFormatNameUserThumbnailMedium = @"com.mycompany.myapp.XXImageFormatNameUserThumbnailMedium";
    static NSString *XXImageFormatFamilyUserThumbnails = @"com.mycompany.myapp.XXImageFormatFamilyUserThumbnails";
    
    FICImageFormat *smallUserThumbnailImageFormat = [[FICImageFormat alloc] init];
    smallUserThumbnailImageFormat.name = XXImageFormatNameUserThumbnailSmall;
    smallUserThumbnailImageFormat.family = XXImageFormatFamilyUserThumbnails;
    smallUserThumbnailImageFormat.style = FICImageFormatStyle16BitBGR;
    smallUserThumbnailImageFormat.imageSize = CGSizeMake(50, 50);
    smallUserThumbnailImageFormat.maximumCount = 250;
    smallUserThumbnailImageFormat.devices = FICImageFormatDevicePhone;
    smallUserThumbnailImageFormat.protectionMode = FICImageFormatProtectionModeNone;
    
    FICImageFormat *mediumUserThumbnailImageFormat = [[FICImageFormat alloc] init];
    mediumUserThumbnailImageFormat.name = XXImageFormatNameUserThumbnailMedium;
    mediumUserThumbnailImageFormat.family = XXImageFormatFamilyUserThumbnails;
    mediumUserThumbnailImageFormat.style = FICImageFormatStyle32BitBGRA;
    mediumUserThumbnailImageFormat.imageSize = CGSizeMake(100, 100);
    mediumUserThumbnailImageFormat.maximumCount = 250;
    mediumUserThumbnailImageFormat.devices = FICImageFormatDevicePhone;
    mediumUserThumbnailImageFormat.protectionMode = FICImageFormatProtectionModeNone;
    
    NSArray *imageFormats = @[smallUserThumbnailImageFormat, mediumUserThumbnailImageFormat];
    

이미지 포멧의 스타일은 아래 4가지가 있다:

  * 32비트 컬러+알파 (default)
  * 32비트 컬러
  * 16비트 컬러
  * 8비트 흑백

소스 이미지에 투명이 없다면(jpeg같이) 32비트 컬러 with no alpha를 사용하면 코어 애니메이션의 퍼포먼스가 향상된다. 소스 이미지가 매우 작거나 컬러가 몇 개 없다면 16비트 컬러를 사용할 수 있다.

#### Configuring the Image Cache

이미지 포멧이 정의되면, 이미지 캐쉬에 할당해야 한다. 델리게이트를 설저아고 이미지 포멧만 넣으면 된다.
    
    
    FICImageCache *sharedImageCache = [FICImageCache sharedImageCache];
    sharedImageCache.delegate = self;
    sharedImageCache.formats = imageFormats;
    

#### Creating Entities

엔티티는 `FICEntity` 프로토콜을 따르는 오브젝트다. 엔티티들은 각각 이미지 테이블의 이미지를 정의하고, 이미지 캐쉬에 저장되어 있는 이미지를 화면에 렌더링한다. 

…

사용법이 꽤나 복잡해 보여, 이것보다는 [SDWebImage](https://github.com/rs/SDWebImage)를 고려하기로 하고 이 포스트는 이쯤에서 정리한다. 실제 적용하려면 원문의 튜토리얼을 따라가도록 하자.

* * *

  1. png, jpg 같은 이미지 포멧이 아니라 앱 내부적으로 사용하는 포멧을 의미↩

  2. recently, least-accessed로 이해하면 편하다. 적게 사용된 이미지를 버린다는 건데 이 때 최근 사용 기록만 보겠다는 것.↩


[Tistory 원문보기](http://khanrc.tistory.com/87)
