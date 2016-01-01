---
layout: post
title: "네트워크 기초 - Switch/Router, LAN/WAN, IP/MAC"
tags: ['Server(Back-end)']
date: 2014-08-05 08:41:00
---
# 네트워크 기초 - Switch/Router, LAN/WAN, IP/MAC

## Switch

L2레이어에서 동작한다. MAC을 보고 포워딩  
하드웨어 레벨

## Router

L3레이어에서 동작. IP를 보고 포워딩  
소프트웨어 레벨

L3-Switch가 라우터래.

* * *

WAN과 WAN을 연결하는게 라우터. 하나의 네트워크 안에서 컴퓨터들을 연결하는게 스위치.  
이건거 같다.

* * *

## [LAN, WAN, VPN, PAN, MAN](http://www.techopedia.com/2/29090/networks/lanwanman-an-overview-of-network-types)

  * LAN  
Local Area Network

  * WAN  
Wide Area Network

  * [VPN](https://mirror.enha.kr/wiki/%EA%B0%80%EC%83%81%20%EC%82%AC%EC%84%A4%EB%A7%9D)  
Virtual Private Network  
전용선으로 다 연결하여 인트라넷을 구축하는 기존의 방법은, 회사가 여기저기 퍼져있으면 비용이 많이 들 뿐만 아니라 외부에서 접근이 원천적으로 불가능하여 불편함이 많다. 따라서 기존 인터넷 망을 이용하면서 암호화된 프로토콜을 통해 사설망을 제공하는 VPN이 등장하였다.

  * [PAN](http://ko.wikipedia.org/wiki/%EA%B0%9C%EC%9D%B8_%ED%86%B5%EC%8B%A0%EB%A7%9D)  
Personal Area Network. 한 사람의 범위 내에서 여러 컴퓨터 기기들(컴퓨터, 노트북, 핸드폰, 태블릿 등) 간의 통신을 위한 네트워크.

  * [MAN](http://www.terms.co.kr/MAN.htm)  
Metropolitan Area Network. 큰 도시나 캠퍼스에 퍼져 있는 네트워크. LAN과 WAN의 중간 크기이다.

## [Ethernet](http://www.terms.co.kr/Ethernet.htm)

LAN 내에서의 근거리통신망 기술

## [MAC Address](http://ko.wikipedia.org/wiki/MAC_%EC%A3%BC%EC%86%8C)

네트워크 어댑터1에 부착된 Unique Identifier. Media Access Control의 약자.  
IP는 MAC과 대응되는 '주소'의 개념이고, MAC은 물리적 유니크 아이덴티파이어다. 즉 IP는 MAC을 찾는 도구로 사용된다. 택배를 보낼 때 주소를 쓰면 택배원이 그 주소로 위치를 찾아가지만, 결과적으론 이름으로 상대가 맞는지 인증한다. 이때 주소가 IP고, 이름이 MAC이다. 이때 IP로 MAC을 알아내는 프로토콜을 ARP(Address Resolution Protocol)이라 한다.

IP는 OSI 3계층인 네트워크 레이어에서 동작하고, MAC은 OSI 2계층인 데이터링크 레이어에서 동작한다. IP주소를 이용하여 상대 위치가 어딘지 알고, MAC을 이용해서 데이터링크 통신을 하는 것.

<http://seeit.kr/317>  
<http://doo8866.tistory.com/entry/MAC-Address-%EB%9E%80>  
[http://www.netmanias.com/ko/?m=view&amp;id=qna&amp;no=4672](http://www.netmanias.com/ko/?m=view&id=qna&no=4672)

* * *

  1. 네트워크 카드(Network card), 랜 카드, 네트워크 인터페이스 카드(NIC), 이더넷 카드라고도 불린다. 피지컬 레이어와 데이터링크 레이어 장치를 가지는데, MAC address를 사용하여 로우레벨 어드레스 시스템을 제공하며 네트워크 매개체로써 물리적인 접근을 가능하게 한다. ↩


[Tistory 원문보기](http://khanrc.tistory.com/16)
