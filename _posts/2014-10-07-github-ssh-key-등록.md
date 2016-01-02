---
layout: post
title: "github: ssh key 등록"
tags: ['ETC']
date: 2014-10-07 01:00:00
---
# github: ssh key 등록

어떤 컴퓨터에서 `github`에 접근하려면 미리 컴퓨터를 등록해야 한다. 자세한 과정은 [GitHub: Generating SSH keys](https://help.github.com/articles/generating-ssh-keys/)를 참고하자.

만약 이러한 등록을 미리 진행하지 않고 `github`에 접근하려고 하면
    
    
    Permission denied (publickey).
    

라는 메시지와 함께 접근을 거부당하게 된다.

`ssh -T git@github.com` 라는 명령어를 통해 인증이 되었는지 아닌지를 확인할 수 있다.
    
    
    $ ssh -T git@github.com
    Hi khanrc! You've successfully authenticated, but GitHub does not provide shell access.
    

인증이 되면 이렇게 반갑게 맞아준다.

참고로 `ssh -vT git@github.com` 를 사용하면 인증 과정도 볼 수 있는 것 같다.

## 참고

[GitHub: Generating SSH keys](https://help.github.com/articles/generating-ssh-keys/)  
[GitHub: SSH](https://help.github.com/categories/ssh/)  
ssh키 사용에 대한 상세한 안내가 되어 있다. 영어 문서라 가슴아프지만 가능하면 읽어보도록 하자.


[Tistory 원문보기](http://khanrc.tistory.com/50)
