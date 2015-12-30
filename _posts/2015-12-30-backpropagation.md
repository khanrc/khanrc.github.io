---
layout: post
title:  "Backpropagation"
categories: [design, tools]
tags: [tag1, tag2]
---

# Backpropagation

https://www.youtube.com/watch?v=q0pm3BrIUFo
2010년 가을학기, MIT AI 수업 Patrick Winston 교수님 강의. 

지금까지 봤던 Backpropagation 강의 중에서 가장 명쾌하다. 중간즈음부터 보면 됨.

가장 핵심적인 스크린샷 몇 장을 소개한다:

- - -
![bp1]({{ site.url }}/assets/bp.png)
![bp2](/assets/bp2.png)
![bp3](/assets/bp3.png)

- - -  

자세한 건 강의를 참고하자. 

백프로파게이션 알고리즘의 핵심은, 멀티레이어 뉴럴 네트워크를 "효율적" 으로 학습할 수 있게 해 준다는 것이다. 이전 레이어로 넘어갈 때마다 모든 연산을 다시 계산해야 할 필요가 없고, 이전 레이어에서 계산한 내용을 재활용 할 수 있기 때문에 매유 효율적으로 연산이 가능하다.
