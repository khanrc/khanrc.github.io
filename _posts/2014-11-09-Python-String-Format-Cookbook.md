---
layout: post
title: "Python String Format Cookbook"
tags: ['Python']
date: 2014-11-09 13:27:00
---
# [Python String Format Cookbook](https://mkaz.com/2012/10/10/python-string-format/)

자세한 건 위 링크를 참고.

## Order % style

`Python 2.6`에서 [str.format()](https://docs.python.org/2/library/stdtypes.html#str.format)이 등장했다. 이전에는 %를 이용해서 출력했는데 이러한 방식의 단점을 보완한 것 같다. 이전 방식은 아래와 같다:
    
    
    >>> "%s %s" % ("hi", "hoi")
    'hi hoi'
    

[Stackoverflow: Python string formatting: % vs. .format](http://stackoverflow.com/questions/5082452/python-string-formatting-vs-format) 이 링크에 가 보면 %방식의 단점을 알 수 있다. 

## str.format

그래서 str.format()을 소개한다.

### Basic
    
    
    >>> "glazed with {} water beside the {} chickens".format("rain", "white")
    'glazed with rain water beside the white chickens'
    >>> " {0} is better than {1} ".format("emacs", "vim")
    ' emacs is better than vim '
    >>> " {1} is better than {0} ".format("emacs", "vim")
    ' vim is better than emacs '
    

이렇게 그냥 차례대로 쓸 수도, 순서를 지정할 수도 있다.

### Not Enough Arguments
    
    
    >>> " ({0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}) ".format(1,2,3,4,5,6,7,8,9,10)
    ' (1, 2, 3, 4, 5, 6, 7, 8) '
    >>> " ({}, {}, {}, {}, {}, {}, {}, {}) ".format(1,2,3,4,5,6,7,8,9,10)
    ' (1, 2, 3, 4, 5, 6, 7, 8) '
    

Order % style과 다르게 개수가 서로 맞지 않아도 상관없다. 물론, 출력하려고 하는 변수는 다 있어야 출력이 가능하다.

### Named Arguments
    
    
    >>> " I {verb} the {object} off the {place} ".format(verb="took", object="cheese", place="table")
    ' I took the cheese off the table '
    

이렇게 이름을 지정할 수도,

### Reuse Same Variable
    
    
    >>> "Oh {0}, {0}! wherefore art thou {0}?".format("Romeo")
    'Oh Romeo, Romeo! wherefore art thou Romeo?'
    

이렇게 하나의 변수를 여러번 출력할 수도 있다.

### Base Conversion
    
    
    >>> "{0:d} - {0:x} - {0:o} - {0:b} ".format(21)
    '21 - 15 - 25 - 10101 '
    

이렇게 형변환도 가능하다. 차례로 10진수, 16진수, 8진수, 2진수.

### Use Format as a Function
    
    
    >>> email_f = "Your email address was {email}".format
    >>> email_f(email="bob@example.com")
    'Your email address was bob@example.com'
    

무려 이런 것도 가능하다. 함수로 만들어서 활용한다.

### Escaping Braces
    
    
    >>> " The {} set is often represented as {{0}} ".format("empty")
    ' The empty set is often represented as {0} '
    

{ }를 출력하고 싶다면? 두번 써 주자.

### Number Formatting

NUMBER | FORMAT | OUTPUT | DESCRIPTION  
---|---|---|---  
3.1415926 | {:.2f} | 3.14 | 2 decimal places  
3.1415926 | {:+.2f} | +3.14 | 2 decimal places with sign  
-1 | {:+.2f} | -1.00 | 2 decimal places with sign  
2.71828 | {:.0f} | 3 | No decimal places  
5 | {:0&gt;2d} | 05 | Pad number with zeros (left padding, width 2)  
5 | {:x&lt;4d} | 5xxx | Pad number with x's (right padding, width 4)  
10 | {:x&lt;4d} | 10xx | Pad number with x's (right padding, width 4)  
1000000 | {:,} | 1,000,000 | Number format with comma separator  
0.25 | {:.2%} | 25.00% | Format percentage  
1000000000 | {:.2e} | 1.00e+09 | Exponent notation  
13 | {:10d} |           13 | Right aligned (default, width 10)  
13 | {:&lt;10d} | 13 | Left aligned (width 10)  
13 | {:^10d} |      13 | Center aligned (width 10)  
  
  



[Tistory 원문보기](http://khanrc.tistory.com/70)
