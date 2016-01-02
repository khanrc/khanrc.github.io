---
layout: post
title: "Move To Jekyll"
tags: ['ETC']
---

블로그를 티스토리에서 Jekyll을 이용한 Github pages로 옮겼다. 예전부터 옮겨야겠다고 생각은 했는데, 바쁘기도 했거니와 내가 웹 프론트엔드에 경험이 없다보니 시작하기가 조금 힘들었던 것도 있다. 옮기고 나니 티스토리에 비해 아쉬운 점이 좀 눈에 띄어서 그냥 티스토리를 쓸까 잠깐 생각도 해 봤는데, 결국 장기적으로는 자유도가 높은 지킬로 가는게 맞는 것 같다.

### Jekyll vs Lektor
최근에 플라스크 (flask) 의 개발자가 Lektor라는 파이썬 기반의 정적 사이트 생성기를 만들었다. 나는 아무래도 루비도 잘 모르고 파이썬을 좋아하기 때문에 렉터를 쓸까 고민을 좀 했었는데 결국 지킬을 선택했다. 가장 큰 이유는 지킬 커뮤니티가 훨씬 크다는 것이다. 내가 웹 프론트엔드에 능숙하면 큰 상관 없겠지만 그렇지 않기 때문에 문제가 생겼을 때 참고할 자료가 많아야 했다.

### [Redcarpet vs Kramdown](https://gist.github.com/mikelove/cbf6eb431406852ba725)
마크다운을 html로 변환해주는 엔진이다. 둘 다 많이 쓰는 엔진이지만 제각각 문제가 있어서 뭘 선택할 지 한참 고민했다. Redcarpet은 mathjax를 쓸 때 manual escaping을 해 줘야 한다. tex 코드를 그대로 입력하면 파싱이 제대로 안 될 때가 있다. kramdown은 그 부분에선 괜찮은데 다른 문자들에 대한 escaping 이슈가 있다. 대표적으로 pipe 문자 (|) 가 문제인데, 파이프 문자는 GFM 에서 테이블을 표현하기 위해 쓰여서 kramdown은 이 파이프 문자를 시도때도 없이 테이블로 만들어 버린다. 즉, 파이프 문자를 쓰고 싶으면 escaping을 해 줘야 한다.

나는 고민 끝에 Redcarpet으로 정했다. mathjax 가 제대로 작동을 안 해서 tex 코드가 그대로 나오는 것까진 참을 수 있지만 중간에 이상하게 테이블이 들어가 버리면 아예 해독이 안 될 수 있다. 그리고 레드카펫의 no_intra_emphasis 옵션도 매력적이다.

### [Migration](https://github.com/khanrc/tistory2jekyll)
티스토리의 글들을 마크다운으로 변환하기 위해 마이그레이터를 만들었다. html을 마크다운으롤 변환하기 위한 파서로는 html2text 를 사용했다. 위에서 말한 것처럼 mathjax 에 문제가 있고, category 시스템에서 tag 시스템으로 변경하면서 수정해 주고 싶은 부분도 있었기 때문에 일괄 처리를 위해 rearranger를 만들었다. rearranger는 태그를 변경하고, mathjax 코드를 div태그로 묶어서 깨지지 않게 돕는다.



