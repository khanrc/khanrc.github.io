---
layout: post
title: "RL - Meta-RL"
tags: ['RL']
date: 2019-04-01
comments: true
---

* TOC
{:toc}

# Meta-RL

추천 레퍼런스: [cs294 - Meta RL slides (Chelsea Finn)](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-20.pdf)

## Meta-Learning

- Meta learning 이란
    + Basic method: fine-tuning (transfer learning)
        * 이것보다 더 잘하고 싶다면?
    + One/Few-shot learning
- Meta learning 의 분류
    + Model-based
    + Metric-based
    + Optimization-based
- 이러한 Meta-learning 은 RL 에도 그대로 적용할 수 있다

## RL^2

Duan, Yan, et al. "RL $^ 2$: Fast Reinforcement Learning via Slow Reinforcement Learning." arXiv preprint arXiv:1611.02779 (2016).

## SNAIL

Mishra, Nikhil, et al. "A simple neural attentive meta-learner." arXiv preprint arXiv:1707.03141 (2017).

SNAIL (Simple Neural Attentive Learner)

## MAML

Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.

추천 레퍼런스: [PR12-094: MAML](https://youtu.be/fxJXXKZb-ik)

MAML (Model-Agnostic Meta-Learning) 은 meta learning 이라는 task 를 그대로 모델링한다. 