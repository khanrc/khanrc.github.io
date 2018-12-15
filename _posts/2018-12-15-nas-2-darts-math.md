---
layout: post
title: "AutoML (3) - DARTS: math"
tags: ['AutoML']
date: 2018-12-15 01:00:00
---

- https://www.facebook.com/sungbin87/posts/2109315539093116

## Method

### Algorithm

<img src="{{site.url}}/assets/nas/2-darts-algo.png">

우리의 모델은 child network weight $w$ 뿐만이 아니라 architect parameter $\alpha$ 도 갖는다. 거기다가 우리는 이 $\alpha$ 에 대해서는 unrolled gradient 를 계산해야 하기 때문에 update 2 가 상당히 복잡해진다. 먼저 식을 정리해보자.

### Preliminary

Unrolled gradient 를 계산하기 위해 알아야 할 것이 두 가지 있다.

#### The Multivariable Chain Rule

https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/differentiating-vector-valued-functions/a/multivariable-chain-rule-simple-version

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

#### Taylor series

http://darkpgmr.tistory.com/59

https://en.wikipedia.org/wiki/Taylor%27s_theorem

https://en.wikipedia.org/wiki/Taylor_series

임의의 함수 f(x) 가 x=a 에서 무한 번 미분 가능하다면,

$$
\begin{align}
f(x)&=f(a)+f'(a)(x-a)+\frac{f''(a)}{2!}(x-a)^2+\cdots \\
&=\sum^\infty_{k=0} \frac{f^{(k)}(a)}{k!}(x-a)^k
\end{align}
$$

가 x=a 근처에서 성립한다.

여기서 x,a 를 a, a+h 로 바꿔 표기하는 방식도 자주 쓰인다.

$$
\begin{align}
f(a+h)&=f(a)+f'(a)h+\frac{f''(a)}{2!}h^2+\cdots \\
&=\sum^\infty_{k=0} \frac{f^{(k)}(a)}{k!}h^k
\end{align}
$$

보통 second-order derivatives 까지 사용해서 근사 식으로 많이 쓰이는 듯 (2차 근사).

##### Intuition

테일러 급수로 근사한 식을 p(x) 라 한다면 ($f(x)=p_\infty(x)$), f'(a)=p'(a), f''(a)=p''(a), ... 이 성립한다. 즉, 테일러 급수는 x=a 에서 동일한 미분계수를 갖는 함수로 근사하는 방법. 위키를 참고하면 실제로 근사가 진행되면서 (고차 미분이 더해지면서) 점점 정확해지는 것을 볼 수 있다.

##### Taylor's theorem

Unrolled gradient 를 이해하기 위해 필요한 것은 아니나 알아두자.

테일러 정리로부터,

$$
\begin{align}
f(x)&=f(a)+f'(a)(x-a)+ \cdots +\frac{f^{(n)}(a)}{n!}(x-a)^n+R_{n+1}(x) \\
&=\sum^n_{k=0} \frac{f^{(k)}(a)}{k!}(x-a)^k + R_{n+1}(x)
\end{align}
$$

라 할 수 있고, 여기서 n차 approximation error $R_{n+1}(x)$ 에 대해:

$$
R_{n+1}(x)=\frac{f^{(n+1)}(b)}{(n+1)!}(x-a)^{n+1}
$$

을 만족하는 $b \in [x,a] \text{ or } [a,x]$ 가 존재한다.

#### Finite Difference Approximation of Derivatives

https://www.rose-hulman.edu/~bryan/lottamath/diffgrad.pdf

https://www.geophysik.uni-muenchen.de/~igel/Lectures/NMG/02_finite_differences.pdf

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

### Unrolled gradient

> 주의) 여기서 $\nabla f(x) = (\nabla f)(x)$ 다. 만약 $\nabla (f(x))$ 의 표기가 필요할 경우 따로 표기. 논문에서도 이러한 방식을 따랐으나 $\nabla (f(x))$ 방식의 표기는 쓰지 않았음 (쓸 일이 없었음).
>
> 아래 임성빈 박사님의 표기가 더 명확한 표기로 혼동이 없으나 여기서는 논문의 방식을 따름.

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

가 되어, 연산 복잡도가 $O(|\alpha||w|)$ 에서 $O(|\alpha|+|w|)$ 로 줄어든다. 또한 여기서 추가로 등장하는 두 하이퍼파라메터 $\xi$ 와 $\epsilon$ 에 대한 실험적인 설정값도 논문에서 (경험적으로) 제공한다. virtual gradient step 의 learning rate 인 $\xi$ 는 $w$ 의 learning rate 와 동일하게 사용하며, 엡실론의 경우 $\epsilon=0.01 / ||\nabla_{w'} L_{val}(w',\alpha)||$ 를 사용한다. 

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

#### Hessian term - eq (7)

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

가 된다. 여기서 양변에 $\epsilon \nabla_{w'} L_{val}(w',\alpha)$ 를 곱해주면:

$$
\nabla^2_{\alpha,w} L_{train}(w,\alpha) \nabla_{w'} L_{val} (w', \alpha) \approx \frac{\nabla_\alpha L_{train}(w^+,\alpha) - \nabla_\alpha L_{train}(w^-,\alpha)}{2\epsilon}
$$

즉 식 (7) 이 나온다! 즉, 그냥 finite difference approximation 을 수행하되, 우리가 원하는 형태로 나오도록 h 의 값을 맞춰준 것 뿐이다.

아래 임성빈 박사님의 계산에서는 2차 테일러 근사를 사용하니 참고하자:



관련 증명:

- https://www.facebook.com/photo.php?fbid=1985180451506626&set=a.361342977223723&type=3&theater

> 식 (6)은 chain rule 로 간단하게 유도할 수 있습니다. 한 편 식 (7)은 finite difference method 에 익숙한 분은 아실테지만 2차 Taylor approximation 을 사용해서 도출할 수 있습니다.

<img src = "{{site.url}}/assets/nas/3-darts-lsb-prove.jpg" width=70%>