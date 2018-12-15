---
layout: post
title: "AutoML (3) - DARTS: math"
tags: ['AutoML']
date: 2018-12-15 01:00:00
---

# Mathmatics in DARTS

## Algorithm

<img src="{{site.url}}/assets/nas/2-darts-algo.png">

우리의 모델은 child network weight $w$ 뿐만이 아니라 architect parameter $\alpha$ 도 갖는다. 거기다가 우리는 이 $\alpha$ 에 대해서는 unrolled gradient 를 계산해야 하기 때문에 update 2 가 상당히 복잡해진다. 먼저 식을 정리해보자.

## Preliminary

Unrolled gradient 를 계산하기 위해 알아야 할 것이 두 가지 있다.

### The Multivariable Chain Rule

다변수 함수의 미분. 정확히 말하면 다변수 함수의 미분은 보통 우리가 하던 partial derivative 로 하는거고, 여기서 하는 다변수 함수의 미분이란 합성함수의 미분에 가깝다. 각 변수들이 사실은 하나의 변수의 함수인 것.

Single variable:

$$
\frac{d}{dt}f(x(t))=\frac{\partial f}{\partial x}\frac{\partial x}{\partial t}
$$

Multi variable:

$$
\frac{d}{dt}f(x(t),y(t))=
\frac{\partial f}{\partial x}\frac{\partial x}{\partial t} + 
\frac{\partial f}{\partial y}\frac{\partial y}{\partial t}
$$

### Finite Difference Approximation of Derivatives

Finite difference 는 사실 그냥 numerical gradient 다. 미분의 정의에 의해

$$
\nabla_xf=\lim_{h \rightarrow 0} \frac{f(x+h)-f(x-h)}{2h}
$$

가 되는데, 여기서 limit 를 빼고 작은 h 에 대해 계산하여 이 값을 근사할 수 있다.

$$
\nabla_x f \approx \frac{f(x+h)-f(x-h)}{2h}
$$

이를 centered difference 라 하고, f(x+h)-f(x) 로 계산하면 forward difference, f(x)-f(x-h) 로 계산하면 backward difference 라 한다.

이 근사의 에러를 계산하기 위해 테일러 급수를 활용할 수 있는데, 이 부분은 위 링크를 참조하자.

## Unrolled gradient

> 주의) 여기서 $\nabla f(x) = (\nabla f)(x)$ 다. 만약 $\nabla (f(x))$ 의 표기가 필요할 경우 따로 표기. 혼동의 소지가 있으나 논문과 동일한 방식을 따름.

update 2 의 식:

$$
L_{val}(w-\xi \nabla_w L_{train}(w,\alpha), \alpha)
$$

에서, $w'= w-\xi \nabla_w L_{train}(w,\alpha)$ 라 두고 $\alpha$ 에 대해 multivariable chain rule 을 통해 그라디언트를 계산하면:

$$
\nabla_\alpha\left[ L_{val}(w',\alpha) \right] =
\nabla_\alpha L_{val}(w',\alpha) 
- \xi \nabla^2_{\alpha,w} L_{train}(w,\alpha) \nabla_{w'} L_{val} (w', \alpha)
\tag{6}
$$

가 된다. 여기서 이 뒤의 헤시안 항을 finite difference 로 근사할 수 있는데:

$$
\begin{align}
\nabla^2_{\alpha,w} L_{train}(w,\alpha) \nabla_{w'} L_{val} (w', \alpha) &\approx \frac{\nabla_\alpha L_{train}(w^+,\alpha) - \nabla_\alpha L_{train}(w^-,\alpha)}{2\epsilon} \tag{7} \\\\
\text{where} \quad
&w^+=w+\epsilon \nabla_{w'} L_{val}(w',\alpha) \\
&w^-=w-\epsilon \nabla_{w'}L_{val}(w',\alpha) \\
&\epsilon=\text{small scalar}
\end{align}
$$

가 되어, 연산 복잡도가 $O(\|\alpha\|\|w\|)$ 에서 $O(\|\alpha\|+\|w\|)$ 로 줄어든다. 또한 여기서 추가로 등장하는 두 하이퍼파라메터 $\xi$ 와 $\epsilon$ 에 대한 실험적인 설정값도 논문에서 (경험적으로) 제공한다. virtual gradient step 의 learning rate 인 $\xi$ 는 $w$ 의 learning rate 와 동일하게 사용하며, 엡실론의 경우 $\epsilon=0.01 / \|\|\nabla_{w'} L_{val}(w',\alpha)\|\|$ 를 사용한다. 

그렇다면, 최종적으로 unrolled gradient (virtual step gradient) 식을 정리해 보자.

$$
\begin{align}
\nabla_\alpha \left[ L_{val}(w-\xi \nabla_w L_{train}(w,\alpha), \alpha) \right] =& \nabla_\alpha L_{val}(w',\alpha) - \xi \nabla^2_{\alpha,w} L_{train}(w,\alpha) \nabla_{w'} L_{val} (w', \alpha) \\
\approx& \nabla_\alpha L_{val}(w',\alpha) - \frac{\nabla_\alpha L_{train}(w^+,\alpha) - \nabla_\alpha L_{train}(w^-,\alpha)}{2\epsilon}
\end{align}
$$

이제 전부 다 넣어서 풀면 다음과 같다:

$$
\nabla_\alpha L_{val}(w',\alpha) - \frac{\nabla_\alpha L_{train}(w+\epsilon \nabla_{w'} L_{val}(w',\alpha),\alpha) - \nabla_\alpha L_{train}(w-\epsilon \nabla_{w'} L_{val}(w',\alpha),\alpha)}{2*0.01 / ||\nabla_{w'} L_{val}(w',\alpha)||}
$$

### Hessian term - eq (7)

헤시안 항을 근사하는 부분이 어려워 보이지만 별 거 없다. Centered finite difference 식

$$
\nabla_x f(x) \approx \frac{f(x+h)-f(x+h)}{2h}
$$


에서 각각 

$$
\begin{align}
f &\leftarrow \nabla_\alpha L_{train}(w,\alpha) \\
x &\leftarrow w \\
h &\leftarrow \epsilon \nabla_{w'} L_{val}(w',\alpha) \\
\end{align}
$$

를 대입하면:

$$
\nabla_w \left[ \nabla_\alpha L_{train}(w,a) \right]
\approx \frac{\nabla_\alpha L_{train}(w^+,a)-\nabla_\alpha L_{train}(w^-,a)}{2\epsilon \nabla_{w'} L_{val}(w',\alpha)}
$$

가 된다. 여기서 양변에 $\nabla_{w'} L_{val}(w',\alpha)$ 를 곱해주면:

$$
\nabla^2_{\alpha,w} L_{train}(w,\alpha) \nabla_{w'} L_{val} (w', \alpha) \approx \frac{\nabla_\alpha L_{train}(w^+,\alpha) - \nabla_\alpha L_{train}(w^-,\alpha)}{2\epsilon}
$$

즉 식 (7) 이 나온다! 즉, 그냥 finite difference approximation 을 수행하되, 우리가 원하는 형태로 나오도록 h 의 값을 맞춰준 것 뿐이다.

## Reference

- https://www.facebook.com/sungbin87/posts/2109315539093116
- https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/differentiating-vector-valued-functions/a/multivariable-chain-rule-simple-version
- https://www.rose-hulman.edu/~bryan/lottamath/diffgrad.pdf
- https://www.geophysik.uni-muenchen.de/~igel/Lectures/NMG/02_finite_differences.pdf
