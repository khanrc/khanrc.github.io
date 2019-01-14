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

### Taylor Series

임의의 함수 $f(x)$ 가 $x=a$ 에서 무한 번 미분 가능하다면,
$$
\begin{align}
f(x)&=f(a)+f'(a)(x-a)+\frac{f''(a)}{2!}(x-a)^2+\cdots \\
&=\sum^\infty_{k=0} \frac{f^{(k)}(a)}{k!}(x-a)^k
\end{align}
$$
가 $x=a$ 근처에서 성립한다.

#### Intuition

테일러 급수로 근사한 식을 p(x) 라 한다면 ($f(x)=p_\infty(x)$), $f'(a)=p'(a)$, $f''(a)=p''(a)$, ... 이 성립한다. 즉, 테일러 급수는 x=a 에서 동일한 미분계수를 갖는 함수로 근사하는 방법. [위키](https://en.wikipedia.org/wiki/Taylor_series)를 참고하면 실제로 근사가 진행되면서 (고차 미분이 더해지면서) 점점 정확해지는 것을 볼 수 있다.

#### Multivariable function

위 식을 다변수 함수로 확장하면, n차 테일러 급수는:
$$
T^{(n)}[\mathbf{f}, \mathbf{a}](\mathbf x)=\sum_{k=0}^n\frac{(\partial^k_\mathbf x \mathbf f)(\mathbf a)}{k!}(\mathbf x - \mathbf a)^k
$$
라 쓸 수 있다. 

> 위 식은 [임성빈 박사님의 포스트](https://www.facebook.com/sungbin87/posts/2109315539093116) 와 [Taylor series for multivariable function](https://www.researchgate.net/publication/286625914_Taylor_Series_For_Multi-Variable_Functions) 의 내용을 적당히 버무렸다. 혹시 위 수식이 정확히 어떻게 계산되는지가 헷갈린다면, Taylor series for multivariable function 문서에 상세히 나와 있으니 참고하자.

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

식 (7) 로부터 시작하자:
$$
\nabla^2_{\alpha,w} L_{train}(w,\alpha) \nabla_{w'} L_{val} (w', \alpha) \approx \frac{\nabla_\alpha L_{train}(w^+,\alpha) - \nabla_\alpha L_{train}(w^-,\alpha)}{2\epsilon}
$$
여기서 오른쪽의 분자 항을 테일러 시리즈로 근사할 수 있다. 그러면 이 때
$$
f(w)= \nabla_\alpha L_{train}(w,\alpha)
$$
라 하면,
$$
f(w) \approx T^{(2)}[f,w](x)
$$
가 x=w 근처에서 성립하고,
$$
\begin{align}
T^{(2)}[f, w](x)&=f(w)+\nabla_w f(w)(w-x)+\frac 12 \nabla^2_wf(w) (x-w)^2 \\
&=\nabla_\alpha L_{train}(w,\alpha)+\nabla_w\nabla_\alpha L_{train}(w,\alpha)(x-w)+\frac{1}{2}\nabla^2_w\nabla_\alpha L_{train}(w,\alpha)(x-w)^2
\end{align}
$$
가 된다. 여기서 이 Taylor series 함수에 $w^+$ 와 $w^-$ 를 넣어 빼주면 $|w^+-w|=|w^- -w|$ 이므로, 첫번째와 세번째 항이 사라진다. 그러면:
$$
\begin{align}
T^{(2)}[\nabla_\alpha L_{train}, w](w^+)-T^{(2)}[\nabla_\alpha L_{train}, w](w^-) 
&= \nabla_w \nabla_\alpha L_{train}(w,\alpha)(w^+ - w^-) \\
&= 2\epsilon \nabla^2_{\alpha,w} L_{train}(w,\alpha) \nabla_{w'} L_{val} (w', \alpha)
\end{align}
$$
이므로 식 (7) 을 얻을 수 있다. 마지막 전개는 $w^+ - w^- = 2\epsilon \nabla_{w'} L_{val}(w',\alpha)$ 이기 때문이다.

> 세번째 항 끼리 뺄 때 중간에 Hessian 이 들어가서 다소 복잡하지만 잘 풀어서 빼 보면 사라지는 것을 확인할 수 있다.

가 되어 식 (7) 을 얻을 수 있다!

## Reference

- https://www.facebook.com/sungbin87/posts/2109315539093116
- https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/differentiating-vector-valued-functions/a/multivariable-chain-rule-simple-version
- https://www.rose-hulman.edu/~bryan/lottamath/diffgrad.pdf
- https://www.geophysik.uni-muenchen.de/~igel/Lectures/NMG/02_finite_differences.pdf
