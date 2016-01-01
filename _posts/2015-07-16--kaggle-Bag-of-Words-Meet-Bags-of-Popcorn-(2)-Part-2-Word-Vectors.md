---
layout: post
title: "[kaggle] Bag of Words Meet Bags of Popcorn - (2) Part 2: Word Vectors"
tags: ['DataScience/Text Mining']
date: 2015-07-16 18:57:00
---
# Part 2: Word Vectors

## Code

파트 2의 코드는 [여기서](https://github.com/wendykan/DeepLearningMovies/blob/master/Word2Vec_AverageVectors.py) 확인할 수 있다.

## 들어가기 전에, 윈도우 환경 세팅

따로 명시하지는 않았지만, 나는 이 튜토리얼을 윈도우 환경에서 `anaconda`를 사용해서 돌리고 있다(원문은 맥에서 작성되었다). 지금까지 사용한 패키지들은 전부 아나콘다에 기본적으로 포함되어 있지만, 앞으로 사용할 패키지 중 `gensim`은 그렇지 않다. 아나콘다에 `gensim`을 설치해야 한다. 설치법은 [여기](https://binstar.org/anaconda/gensim)에 나와 있는데, 매우 간단하다.
    
    
    conda install -c https://conda.binstar.org/anaconda gensim

## Introducing Distributed Word Vectors

이번 파트에서는 Word2Vec 알고리즘으로 생성되는 distributed word vector를 사용하는 데에 초점을 맞춘다. 

이번 파트에서 사용하는 코드들은 인텔 i5 윈도우 기반으로 작성되었다. 원문은 듀얼코어 맥북 프로 기반으로 작성되었다. 원문과 환경이 다르기 때문에 조금 왔다갔다 할 수 있다.

[Word2vec](https://code.google.com/p/word2vec/)은 2013년에 Google이 퍼블리쉬한 [distributed representations](http://www.cs.toronto.edu/~bonner/courses/2014s/csc321/lectures/lec5.pdf) 를 단어에 대해 학습하는 뉴럴 네트워크 임플레멘테이션이다. 이전에도 다른 deep or recurrent neural network(RNN) 구조가 제안되었었으나, 모델을 학습하기 위해 필요한 시간이 너무 길다는 문제가 있었다. Word2vec은 이러한 방법들에 비해 훨씬 빨리 학습한다.

Word2Vec은 유의미한 표현(meaningful representation)을 하기 위해 클래스 라벨을 필요로 하지 않는다. 이는 매우 유용한데, 실제 데이터는 대부분이 라벨이 없기 때문이다(unlabeled). 네트워크에 충분한 트레이닝 데이터(수백억개의 단어들)를 넣으면, 네트워크는 아주 흥미로운 특징을 지닌 단어 벡터를 생성한다. 이 단어 벡터에 따라, 비슷한 의미를 가진 단어들은 클러스터를 형성하고, 클러스터들은 단어들의 관계나, 유사도에 따라 배치된다. 그러면 이런 짓이 가능하다: "king - man + woman = queen".

[Google's code, writeup, and the accompanying papers](https://code.google.com/p/word2vec/)를 체크하자. [이 프레젠테이션](https://docs.google.com/file/d/0B7XkCwpI5KDYRWRnd1RzWXQ2TWc/edit)도 도움이 될 것이다. 오리지널 코드는 C지만, 파이썬을 포함해서 많은 다른 언어들로 포팅되었다. C를 쓰는 것도 좋겠지만 조금 까다롭다(수동적으로 헤더파일을 수정하고 컴파일해야 한다).

> 스탠포드의 [Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/syllabus.html) 도 살펴보자. 내용은 좋으나 양이 너무 많다…

스탠포드의 최근의 연구는 [딥러닝을 감정분석에 적용했다](http://nlp.stanford.edu/sentiment/); 코드는 자바로 되어 있다. 그러나, 문장 파싱에 의존하는 그들의 접근법은 임의의 길이의 문단에 간단히 적용할 수 없다.

Distributed word vector는 강력하고 여러 어플리케이션에서 사용할 수 있다. 특히, 단어 예측과 번역에서. 여기에서는, 우리는 이를 감정분석에 적용한다.

## Using word2vec in Python

`gensim` 패키지를 사용하면 word2vec 임플레멘테이션을 사용할 수 있다. [여기](http://rare-technologies.com/word2vec-tutorial/)에 좋은 튜토리얼이 있다.

Word2Vec이 GPU를 사용하지는 않지만, 매우 많은 연산을 필요로 한다. 구글 버전이나 파이썬 버전 둘 다 멀티쓰레딩을 사용한다. 우리의 모델을 적당한 시간 안에 학습시키기 위해서, `cython`이 필요하다. Word2Vec은 `cython`없이도 작동하지만 몇분 걸릴 모델 학습이 며칠이 걸리게 될 수 있다.

### Preparing to Train a Model

이제 핵심으로 들어가보자! 먼저, 파트 1에서 했던 것처럼 데이터를 `pandas`로 읽자. 단, 이번에는 50,000개의 unlabeled 리뷰들을 담고 있는 **unlabeledTrain.tsv** 도 같이 사용한다. Bag of Words 모델을 만들었던 파트 1에서는 unlabeled 데이터가 쓸모없었지만, Word2Vec은 unlabeled 데이터를 사용해서 학습할 수 있으므로, 이제 50,000개의 리뷰를 추가적으로 사용할 수 있다.
    
    
    import pandas as pd
    
    # Read data from files
    train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
    unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
    
    # Verify the number of reviews that were read (100,000 in total)
    print("Read {0} labeled train reviews, {1} labeled test reviews, and {2} unlabeled reviews\n"\
          .format(train["review"].size,  test["review"].size, unlabeled_train["review"].size ))

데이터 클리닝 함수는 파트 1과 비슷하지만 약간 차이가 있다. 먼저, Word2Vec는 문장의 문맥(context)을 고려하여 하이퀄리티 단어 벡터를 생성하기 때문에, stop word를 제거하는 것이 안 좋을 수 있다. 따라서 아래 함수에서 stop word 제거를 옵셔널하게 바꾸었다. 마찬가지의 이유로 숫자도 남겨두는 것이 더 좋을 수 있는데, 이는 독자들이 직접 해보도록 하자.
    
    
    # Import various modules for string cleaning
    from bs4 import BeautifulSoup
    import re
    from nltk.corpus import stopwords
    
    def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #  
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return words

다음으로, 인풋 포멧을 맞추어야 한다. Word2Vec은 각 문장이 단어 list로 구성된 list를 인풋으로 받는다. 즉, 인풋 포멧은 2중 list 구조다.

문단(paragraph)을 문장(sentence)으로 나누는 것은 간단한 작업이 아니다. 자연어(natural language)에는 수많은 변수들이 존재한다. 영어 문장은 "?", "!", """, "." 등 다양한 문자로 끝날 수 있고, 띄어쓰기나 대문자는 별로 신뢰할만한 기준이 되지 못한다. 이러한 이유로, 문장 분리를 위해 `NLTK`의 **punkt** tokenizer를 사용한다. 
    
    
    import nltk.data
    
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    # Define a function to split a review into parsed sentences
    def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words, so this returns a list of lists
        return sentences

이제, 이 함수들을 사용해서 데이터들을 Word2Vec의 인풋에 맞게 정제하자.
    
    
    sentences = []  # Initialize an empty list of sentences
    
    print("Parsing sentences from training set")
    for i, review in enumerate(train["review"]):
        if (i+1) % 1000 == 0:
            print("[training set] {} of {}".format(i+1, train["review"].size))
        sentences += review_to_sentences(review, tokenizer)
    
    print("Parsing sentences from unlabeled set")
    for i, review in enumerate(unlabeled_train["review"]):
        if (i+1) % 1000 == 0:
            print("[unlabeled set] {} of {}".format(i+1, unlabeled_train["review"].size))
        sentences += review_to_sentences(review, tokenizer)

`BeautifulSoup`이 문장에 포함된 URL들에 대해 경고(warning)하겠지만 걱정하지 않아도 된다. 상당히 오래 걸리는 작업이므로 중간중간 진행과정을 출력하도록 했다. 

이제 결과물을 출력해 보고 파트 1과 어떻게 다른지 살펴보자:
    
    
    In[14]: print(len(sentences))
    795538
    In[15]: print(sentences[0])
    ['with', 'all', 'this', 'stuff', 'going', 'down', 'at', 'the', 'moment', 'with', 'mj', 'i', 've', 'started', 'listening', 'to', 'his', 'music', 'watching', 'the', 'odd', 'documentary', 'here', 'and', 'there', 'watched', 'the', 'wiz', 'and', 'watched', 'moonwalker', 'again']
    In[16]: print(sentences[1])
    ['maybe', 'i', 'just', 'want', 'to', 'get', 'a', 'certain', 'insight', 'into', 'this', 'guy', 'who', 'i', 'thought', 'was', 'really', 'cool', 'in', 'the', 'eighties', 'just', 'to', 'maybe', 'make', 'up', 'my', 'mind', 'whether', 'he', 'is', 'guilty', 'or', 'innocent']
    

원문에는 len(sentences)가 85000+ 이라고 되어 있는데 어째선지 여기서는 80000개도 나오지 않는다. `NLTK`의 stop word가 추가된 것으로 짐작해본다. 혹은, 원문이 작성된 시점 이후에 데이터의 변화가 있었을 수도 있다.

지금까지의 소스를 잘 살펴보면 "+="와 "append"가 혼용되는 것을 볼 수 있는데, 이는 두 명령의 기능적 차이 때문이다. 리스트에 변수를 더할 때는 이 두 명령이 동일하게 작동하나, 리스트에 리스트를 더할 때는 달라진다. 이 때 "+="는 리스트의 원소들끼리 합치는 작업이고, "append"는 기존의 리스트에 새로운 리스트를 통째로 하나의 원소로 추가한다.

실제로 코드를 돌려 보면 위 작업이 엄청 오래 걸린다. 계속 코딩을 해 나가면서 위 작업을 수차례에 걸쳐 반복적으로 돌려야 하는데 그러기에는 너무 오랜 시간이다. 이를 `pickle`패키지를 통해 해결할 수 있다. `pickle`패키지는 파이썬의 객체를 통째로 파일에 덤프하고 로드하는 기능을 제공한다.
    
    
    with open("sentences.pickle", "wb") as f:
        import pickle
        pickle.dump(sentences, f)

파일을 "wb"로 열어야 한다는 점을 주의하자! 피클링 한 객체는 바이트이기 때문에 byte를 의미하는 "wb"를 써야 한다. 한 번 이 코드를 실행하고 나면 이제 sentences.pickle 파일이 생기고, 다음부터는 위 전처리 과정들을 처음부터 돌릴 필요 없이 sentences.pickle로부터 불러오면 된다.
    
    
    print("Load sentences from pickle ...")
    with open("sentences.pickle", "rb") as f:
        import pickle
        sentences = pickle.load(f)

불러올 때도 마찬가지로 "rb"를 사용한다.

## Training and Saving Your Model

이제 잘 파싱된 문장들을 갖췄으니, 모델을 학습할 준비가 되었다. 실행시간과 최종 모델의 정확도에 영향을 끼치는 파라메터들의 값을 선택해야 한다. 아래 알고리즘의 자세한 내용은 word2vec [API documentation](http://radimrehurek.com/gensim/models/word2vec.html)과 [Google documentation](https://code.google.com/p/word2vec/)을 참고하자.

  * **Architecture**: 아키텍처 옵션은 skip-gram (default) 와 continuous bag of words가 있다. skip-gram이 미세하게 느리지만 더 좋은 결과를 보여준다.
  * **Training algorithm**: hierarchical softmax (default) 와 negative sampling이 있다. 여기서는, 디폴트가 좋다.
  * **Downsampling of frequent words**: 구글 도큐먼트에서 .00001에서 .001 사이의 값을 추천한다. 여기서는, 0.001에 가까운 값이 좋아 보인다.
  * **Word vector dimensionality**: 많은 특성(feature)은 더 많은 학습시간을 요구하지만, 보통 더 좋은 결과를 낸다(항상 그런것은 아니다). 수십에서 수백 정도가 적당한 값이다; 우리는 300개의 특성을 사용한다.;
  * **Context / window size**: word2vec은 어떤 단어 주변의 단어들, 즉 문맥을 고려해서 해당 단어의 의미를 파악한다. 이 때 얼마나 많은 단어를 고려해야 할까? 10 정도가 hierarchical softmax에 적당하다. 이 값도 어느정도까지는 높을수록 좋다. 
  * **Worker threads**: 패러렐 쓰레드의 수. 컴퓨터마다 다르겠지만, 일반적으로 4~6 정도가 적당하다.
  * **Minimum word count**: meaningful word를 규정하는 최소 word count. 이 수치 미만으로 등장하는 단어는 무시한다. 10에서 100 사이의 값이 적당하다. 우리의 경우, 각 영화가 30번씩 등장하므로, 영화 제목에 너무 많은 의미 부여를 피하기 위해 minimum word count를 40으로 설정하였다. 그 결과로 vocabulary size는 약 15,000개의 단어다.

파라메터를 선택하는 건 쉽지 않지만, 선택하고 나면 바로 Word2Vec 모델을 만들 수 있다.
    
    
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    
    # Initialize and train the model (this will take some time)
    from gensim.models import word2vec
    print("Training model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)
    
    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)
    
    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    

혹시 이 코드를 돌렸을 때 OverflowError: Python int too large to convert to C long 가 난다면, [gensim github issue page에 관련한 쓰레드](%28https://github.com/piskvorky/gensim/issues/321%29)가 올라와 있다. 제일 마지막에 적혀있는 대로 문제를 해결할 수 있다. (해쉬함수를 바꿨으니 작동도 달라질 수 있는데 거기까진 모르겠다)
    
    
    def myhashfxn(obj):
        return hash(obj) % (2 ** 32)
    
    word2vec.Word2Vec(hashfxn=myhashfxn)

듀얼코어 맥북 프로에서, 이 작업은 4개의 워커쓰레드를 돌릴 때 15분 이하로 걸린다. 이는 컴퓨터마다 상당히 다를 수 있다. 다행히도, 우리가 설정한 `logging`모듈이 친절하게 진행 상황을 알려준다.

만약 맥이나 리눅스 환경이라면, 터미널에서 "top"명령어를 통해 패러렐라이징이 잘 동작하는 것을 확인할 수 있다. 윈도우 환경이라면, PowerShell에서 "While(1) {ps | sort -des cpu | select -f 20 | ft -a; sleep 2; cls}"으로 비슷한 결과를 볼 수 있다.
    
    
    # Linux or Mac
    > top -o cpu
    
    # Windows
    > While(1) {ps | sort -des cpu | select -f 20 | ft -a; sleep 2; cls}

이 명령어를 통해 CPU 상태를 확인해 보면, 리스트의 제일 위에 파이썬이 있을 것이다! 4개의 워커를 사용하기 때문에 300-400%의 CPU usage를 보여준다.

![CPU usage](https://kaggle2.blob.core.windows.net/competitions/kaggle/3971/media/Screen%20Shot%202014-08-04%20at%202.01.40%20PM.png)

> 윈도우에서 돌렸을 땐 결과가 영 딴판이었는데, 추후 다시 체크해 보자.

만약 CPU usage가 낮게 나온다면, `cython`이 제대로 동작하고 있지 않은 것이다.

소스의 끝을 보면, save함수를 통해 모델을 저장하는 것을 볼 수 있다. 실제로 실행 후에 파일이 생성되는데, 모델을 새로 트레이닝 하지 않고 이 파일을 로드할 수 있다.
    
    
    model = word2vec.Word2Vec.load(model_name)

myhashfxn을 사용했다면, 로드하기 전에 마찬가지로 해쉬펑션을 정의해 놓아야 한다!

## Exploring the Model Result

자, 그럼 이제 우리가 75,000 개의 트레이닝 리뷰를 통해 생성한 모델을 살펴보자.

"doesnt_match" 함수는 주어진 단어 셋(set) 중에서 가장 비슷하지 않은 단어를 추정한다:
    
    
    In[3]: model.doesnt_match("man women child kitchen".split())
    Out[3]: 'kitchen'

우리 모델이 이 의미 차이를 구분해낸다! man, women, children이 비슷하고 kitchen과는 다르다는 것을 알고 있다. 이제 도시와 나라같은 미묘한 차이도 구분하는지 확인해 보자:
    
    
    In[4]: model.doesnt_match("france england germany berlin".split())
    Out[4]: 'berlin'

우리가 비교적 작은 트레이닝 셋을 사용했기 때문인지, 이 모델은 완벽하지 않다:
    
    
    In[7]: model.doesnt_match("paris berlin london austria".split())
    Out[7]: 'london'

원문에선 "paris"를 찾는데, 위 전처리 단계에서 데이터가 달라졌으므로 이후 모델링 결과도 다른게 당연하다. 아무튼 둘 다 "austria"를 찾아내지 못한다. 

이번엔 "most_similar" 함수를 써 보자. 우리의 모델이 갖고 있는 단어 클러스터(word cluster)를 살펴볼 수 있다.
    
    
    In[9]: model.most_similar("man")
    Out[9]: 
    [('woman', 0.6386926770210266),
     ('guy', 0.4642142355442047),
     ('boy', 0.4619269073009491),
     ('person', 0.4530017077922821),
     ('men', 0.45294448733329773),
     ('lady', 0.44111737608909607),
     ('girl', 0.4240424931049347),
     ('himself', 0.42025846242904663),
     ('son', 0.39986851811408997),
     ('he', 0.3897513747215271)]
    
    In[10]: model.most_similar("queen")
    Out[10]: 
    [('princess', 0.5116116404533386),
     ('latifah', 0.4850308299064636),
     ('victoria', 0.41973501443862915),
     ('widow', 0.39943596720695496),
     ('england', 0.3919205963611603),
     ('bride', 0.3856983184814453),
     ('selena', 0.3763607144355774),
     ('king', 0.3756728768348694),
     ('bee', 0.3743290305137634),
     ('rudolf', 0.3727717995643616)]

"Latifa"가 "Queen"과 비슷하다고 나오는 건 우리 데이터셋을 살펴보면 놀랍지 않다.

또는, 감정분석을 위해서는 이런 걸 찾아보자.
    
    
    In[11]: model.most_similar("awful")
    Out[11]: 
    [('horrible', 0.6424727439880371),
     ('terrible', 0.6269798874855042),
     ('atrocious', 0.5686882734298706),
     ('dreadful', 0.5647668242454529),
     ('laughable', 0.531086266040802),
     ('appalling', 0.526667058467865),
     ('horrid', 0.5050047636032104),
     ('amateurish', 0.5010923743247986),
     ('abysmal', 0.5010562539100647),
     ('horrendous', 0.49665677547454834)]

지금까지 확인해 본 바에 따르면 이 모델은 감정분석을 하기에 충분해 보인다. 적어도 Bag of Words만큼! 하지만 우리가 어떻게 이 팬시한 단어 벡터(fancy distributed word vectors)를 supervised learning에 사용할 수 있을까? 다음 섹션에선 그 부분을 다룬다.


[Tistory 원문보기](http://khanrc.tistory.com/107)
