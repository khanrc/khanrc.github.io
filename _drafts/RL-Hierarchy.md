---
layout: post
title: "RL - Hierarchical RL"
tags: ['RL']
date: 2019-03-24
comments: true
---

* TOC
{:toc}


# Hierarchical RL

RL 이 맞닥뜨리는 주요한 챌린지 중 하나는 sparse reward 다. RL 은 처음에 랜덤 폴리시로 시작한다. 이 말은 곧 랜덤 폴리시로 reward 를 받을 수 있어야 학습이 가능하다는 의미다. 때문에 sparse reward 인 경우, 랜덤 폴리시로 reward 를 받을 수가 없어 학습이 안 된다. 

![montezuma](/assets/rl/hrl-montezuma.png){:width="60%" .center}
*Sparse reward problem 으로 유명한 몬테주마의 복수 게임*

몬테주마의 복수 게임은 몬스터를 피해 열쇠를 먹고 문으로 가야 하는 게임이다(으로 알고 있다). 랜덤 액션 폴리시로 몬스터를 피해서 열쇠를 먹고 다시 몬스터를 피해 문까지 가야 리워드를 받을 수 있는 환경인 것이다. 이러한 sparse reward problem 에서 지금까지 소개된 기존의 알고리즘들은 거의 학습이 안 되는 모습들을 보여왔다.

흠....

## HIRO

## STRAW

