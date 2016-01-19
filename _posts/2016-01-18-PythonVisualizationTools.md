---
layout: post
title: "Python Visualization Tools"
tags: ['DataScience', 'Python']
---

나는 지금까지 matplotlib 만 사용해 왔었는데, 시각화 패키지들이 꽤 많은 듯 하다. 살짝 정리해 본다.

데이터는 기본적으로 pandas 로 다룬다. pandas 가 필수적인 건 아니지만 주로 pandas를 쓰니까.

## Matplotlib?
파이썬 시각화 (visualization) 패키지들의 조상. 매우 강력하지만 다루기 까다롭다. 같은 그래프를 그려도 코드가 길어짐. 대표적으로 [ggplot과 비교한 이 포스트](http://blog.yhathq.com/posts/ggplot-for-python.html) 를 참고하자. 이런 문제를 해결하기 위해 matplotlib 을 래핑하여 간편한 인터페이스를 제공하는 툴들이 있음. 대표적인 예가 Pandas 와 Seaborn 이다. Pandas는 시각화 패키지라고 소개하기는 좀 어렵지만, matplotlib 와 연동하여 간편하게 그래프를 그릴 수 있도록 지원함.

## Visualization tools
* [ggplot](http://ggplot.yhathq.com/)
	* ggplot 은 R의 ggplot2 를 파이썬용으로 포팅한 라이브러리임. gg는 Grammar of Graphics의 약자인 듯. R에서 포팅하여 문법이 좀 non-pythonic 하지만 굉장히 직관적이고 강력함.
	* 프로페셔널한 그래프를 쉽게 만들 수 있음.
	* 마찬가지로 matplotlib 에 기반하여 만들어짐.
	* 다만 아직 개발중이라는 얘기가...
* [Seaborn](http://stanford.edu/~mwaskom/software/seaborn/introduction.html#introduction)
	* Seaborn은 statistical graphics 라고 함. matplotlib 에 기반하여 만들어졌으며 파이썬의 데이터 라이브러리와 강력하게 연동된다. numpy 와 pandas 의 데이터 스트럭쳐를 지원하며 scipy 와 statsmodels 의 통계학적 루틴들을 지원하는 등.
	* 그래프를 훨씬 예쁘게 그릴 수 있음.
	* 데이터의 통계분포, 리니어 리그레션 등을 시각화하기 쉬움.
	* 다양한 빌트인 테마를 지원함.
* [Bokeh](http://www.datasciencecentral.com/profiles/blogs/opensource-python-visualization-libraries)
	* Bokeh 는 D3.js 와 같은 interactive visualization 을 지원하며, 동시에 대용량 데이터까지 커버할 수 있는 퍼포먼스를 목표로 함 (아마 D3.js 는 대용량 데이터 처리가 잘 안되는듯). 
	* 홈페이지에 있는 예제를 눌러봤을때는 그냥 확대축소 정도만 지원하는 것 같은데?
	* 웹브라우저에 적합한 시각화, 즉 interactive web visualization 을 목적으로 함.
* [pygal](http://www.pygal.org/en/latest/)
	* pygal은 dynamic SVG charting library 다. SVG 라는 건 interactive 한 그래프를 그릴 수 있다는거임. 링크를 참조하자. 
	* 또한 더욱 다양한 차트 스타일을 지원하며 테마를 css 를 통해 수정할 수 있다는 듯.
* [python-igraph](https://pypi.python.org/pypi/python-igraph)
	* 얘는 잘 모르겠는데 network analysis 를 할 수 있다는 듯.
	* igraph 의 파이썬 인터페이스라고 함.
* [Plot.ly](https://plot.ly/)
	* online tool 임. 즉, 파이썬으로 코딩을 하면 api 를 호출해서 그래프를 생성해서 url을 리턴함. web-based 인 만큼 interactive graph. 물론 이미지로 저장할수도 있음.
* [mpld3](http://mpld3.github.io/)
	* mpl + d3 (matplotlib + d3js) 인 듯. matplotlib 로 그린 그래프를 웹에 올릴 수 있도록 html code로 바꿔주는 라이브러리.
* [folium](https://github.com/python-visualization/folium)
	* Map visualization tool. [basemap](http://matplotlib.org/basemap/) 이라는 matplotlib 확장 (extension) 을 사용하면 matplotlib 에서도 지도 기반 시각화를 할 수 있는데. folium 은 interactive map visualization을 지원함.
	* leaflet.js 의 python wrapper 라고 하는듯.
* [NetworkX](https://github.com/networkx/networkx)
	* Network visualization tool. [gallery](http://networkx.github.io/documentation/latest/gallery.html) 참고.
* [vispy](vispy.org)
	* 홈페이지에 가 보면 엄청 화려한 그래프들을 보여주는데... 사용하기도 까다로울 것 같음. 마찬가지로 interactive.

## Summary
논문 라이팅을 위해서는 ggplot 이 적합하지 않을까 싶음. 필요할 때 networkxx 도 활용할 수 있을 듯. 나머지는 대부분 interactive visualization 이라서 논문에는 적합하지 않아 보임. 

## References
* [Comparing 7 Python data visualization tools](https://www.dataquest.io/blog/python-data-visualization-libraries/)
* [Overview of Python Visualization Tools](http://pbpython.com/visualization-tools-1.html)
* [Python Visualization Libraries List](http://www.datasciencecentral.com/profiles/blogs/opensource-python-visualization-libraries)






