---
layout: post
title: "Syntax Highlight"
tags: ['일상']
date: 2014-09-12 13:32:00
---
# Syntax Highlight

이 블로그를 뚜렷한 컨셉을 갖고 만들진 않았지만, 아무튼 개발 블로그라고 할 수 있다. 개발 공부하면서 흩어진 잡다한 지식들을 정리하기 위함이 첫번째 목적이었으니까. 그러다 보니 자연스럽게 포스트에 코드가 올라가기 마련이고 코드 신택스 하이라이팅의 필요성을 느꼈다.

[신택스 하이라이트 라이브러리 비교](http://demun.tistory.com/2412)  
highlight.js를 소개하는 글이지만 비교가 잘 되어 있으니 참고하자.

## [Highlight.js](https://highlightjs.org/)

**장점**

  * 자동 언어 감지
  * 적용이 편하다  
하나의 js파일과 하나의 css만으로 적용이 가능하다
  * 가벼움  
코드 용량이 적다
  * 빠름  
로딩 및 신택스 하이라이팅이 빠르다
  * Markdown default  
내가 highlight.js를 사용하는 가장 큰 이유다. 블로깅을 할 때 마크다운을 활용하는데, 마크다운에서 Fenced Code Blcok을 사용하면 highlight.js로 신택스 하이라이팅이 된다. 특히, 자동 언어 감지 기능과 결합되어 예전에 올렸던 코드들도 자동으로 신택스 하이라이팅이 된다는 점은 매우 편리하다.

**단점**

  * 줄번호 표기 불가
  * 그래서인지 별로 안 예뻐 보인다.

[highlight.js를 tistory에서 사용하기.](http://gyuha.tistory.com/456)  
그대로 따라하면 된다.

## [SyntaxHighlighter](http://alexgorbatchev.com/SyntaxHighlighter/)

검색어에 따라 달라지겠지만 구글에 대충 검색해 보면 제일 많이 나오는 것 같다. 실제로 블로그들 돌아다녀 봐도 이게 제일 많이 쓰이는 것으로 보인다. 그래서인지 예쁘다.

**장점**

  * 예쁘다
  * 줄번호 표기

**단점**

  * 무거움
  * 느림
  * 파일이 많다  
티스토리에 파일을 업로드 할 수 있지만 따로 폴더별 관리라던가 세부적인 관리는 불가능하다. 즉, 파일이 많으면 적용 및 관리가 힘들다.

[블로그(티스토리)에 소스 코드 삽입(하이라이트) Syntax Highlighter](http://diveangel.tistory.com/11)


[Tistory 원문보기](http://khanrc.tistory.com/32)
