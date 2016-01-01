---
layout: post
title: "git tracking list에서 파일 제거하기"
tags: ['그 외/Tech']
date: 2015-07-08 19:56:00
---
# git tracking list에서 파일 제거하기

찾아놓고 보니 별것도 아닌데, 그간 이것 때문에 상당히 고생했다 -_-;;

.gitignore를 늦게 만들면, 이미 tracking 하고 있는 파일들은 무시되지 않는다. 즉, tracking list에 남아있다. 이걸 정리하기 위한 방법은 크게 두가지가 있는데, 첫째는 git repository 자체를 날렸다가 새로 파면 .gitignore가 처음부터 적용된다. 근데 이건 너무 무식하니까, 파일을 지정해서 tracking list에서 제거하는 방법이 있다:
    
    
    $ git rm --cached <file>

이러면 트래킹 리스트에서 파일이 사라진다!

참고로 트래킹 파일 리스트는 이렇게 볼 수 있다:
    
    
    $ git ls-files

github의 [Ignoring files](https://help.github.com/articles/ignoring-files/) 참고.


[Tistory 원문보기](http://khanrc.tistory.com/100)
