---
layout: post
title: "[kaggle] Bag of Words Meet Bags of Popcorn - (1) Part 1: Bag of Words"
tags: ['Text Mining']
date: 2015-07-08 11:46:00
---
# Part 1: For Beginners - Bag of Words

### What is NLP?

NLP는 텍스트 문제에 접근하는 테크닉들의 집합이다. 이 페이지에서는 IMDB 영화 리뷰를 로드하고, 클리닝하고, 간단한 [Bag of Words](http://en.wikipedia.org/wiki/Bag-of-words_model)모델을 적용하여 리뷰가 긍정인지 부정인지 예측해본다.

### Code

파트 1의 코드는 [여기서](https://github.com/wendykan/DeepLearningMovies/blob/master/BagOfWords.py) 확인할 수 있다.

### Reading the Data

데이터 페이지에서 필요한 파일들을 받을 수 있다. 일단 25,000개의 IMDB 영화 리뷰가 들어 있는 **unlabeldTrainData.tsv**가 필요하다. 

그럼 이제 탭으로 구분되어 있는(tab-delimited) 파일을 파이썬으로 읽어보자. 이를 위해 **pandas**를 사용한다. 
    
    
    # Import the pandas package, then use the "read_csv" function to read
    # the labeled training data
    import pandas as pd       
    train = pd.read_csv("labeledTrainData.tsv", header=0, \
                        delimiter="\t", quoting=3)

"header=0"은 파일의 첫번째 줄이 컬럼의 이름이라는 것을 나타내고, "delimiter=\t"는 구분자가 탭이라는 것을, "quoting=3"은 쌍따옴표(doubled quote)를 무시하라는 것을 의미한다. quoting을 주지 않으면 데이터를 불러올 때 쌍따옴표를 제거하고 불러온다. 

아래와 같이 데이터를 출력해볼 수 있다.
    
    
    print(train["review"][0])

### Data Cleaning and Text Preprocessing

#### Removing HTML Markup: The BeautifulSoup Package

데이터를 보면 알겠지만 리뷰에 HTML태그가 포함되어 있다. 이를 제거하기 위해 [Beautiful Soup](http://www.crummy.com/software/BeautifulSoup/bs4/doc/)를 사용하자. 
    
    
    # Import BeautifulSoup into your workspace
    from bs4 import BeautifulSoup
    
    # Initialize the BeautifulSoup object on a single movie review
    example1 = BeautifulSoup(train["review"][0])
    
    # Print the raw review and then the output of get_text(), for
    # comparison
    print(train["review"][0])
    print(example1.get_text())

`get_text`함수는 html문서에서 text만 뽑아내는 것으로, 위와 같이 출력해 보면 태그가 삭제된 것을 확인할 수 있다.

#### Dealing with Punctuation, Numbers and Stopwords: NLTK and regular expressions

텍스트를 클리닝 할 때는 우리가 해결하고자 하는 문제가 무엇인지에 대해 생각해 보아야 한다. 많은 문제에서는 구두점(punctuation)을 제거하는 것이 일반적이지만, 이 문제에서는 감정분석 문제를 다루고 있고, "!!!"이나 ":-(" 등이 감정표현이 될 수 있으므로 구두점도 단어로 다뤄야 한다. 그러나 이 튜토리얼에서는 문제의 단순화를 위해 전부 제거할 것이다.

마찬가지로 이 튜토리얼에서는 숫자를 제거할 것이지만, 실제로는 숫자를 다루는 여러 방법이 있다. 숫자 또한 단어로 취급한다거나, 모든 숫자를 "NUM"이라는 플레이스홀더로 대체한다거나.

모든 구두점과 숫자를 제거하기 위해, 정규표현식(regular expression) 패키지 [`re`](https://docs.python.org/2/library/re.html#)를 사용하자. 
    
    
    import re
    # Use regular expressions to do a find-and-replace
    letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                          " ",                   # The pattern to replace it with
                          example1.get_text() )  # The text to search
    print(letters_only)

해보면 모든 구두점들이 공백으로 바뀐 것을 확인할 수 있다.

이제, 각 단어들을 소문자로 바꾸고 단어별로 분리하자. (NLP 에서는 "tokenization"이라 한다)
    
    
    lower_case = letters_only.lower()        # Convert to lower case
    words = lower_case.split()               # Split into words

자 이제 마지막으로, "[stop words](http://en.wikipedia.org/wiki/Stop_words)"라고 불리는 큰 의미가 없는 단어들을 어떻게 다룰 것인지 결정해야 한다. 영어에서 "a", "and", "is", "the" 등이 여기에 속한다. 이를 처리하기 위해 [Natural Language Toolkit](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words) (NLTK)를 사용한다. 참고로, `nltk`는 설치 후 `nltk.download()`를 통해 구성요소들을 다운로드 해 주어야 한다. (하지 않으면 하라고 에러 메시지가 뜬다)
    
    
    import nltk
    # nltk.download()  # Download text data sets, including stop words
    
    from nltk.corpus import stopwords # Import the stop word list
    print(stopwords.words("english"))
    
    # Remove stop words from "words"
    words = [w for w in words if not w in stopwords.words("english")]
    print(words)

`nltk`는 stopwords corpus를 가지고 있다. 이를 불러와서 우리의 리뷰 데이터에서 stopword를 제거해주자. example1, 즉 첫번째 리뷰의 단어가 437개에서 222개로 절반 가까이 줄어드는 것을 확인할 수 있다!

텍스트 데이터에 할 수 있는 더 많은 것들이 있는데, 예를 들면 Porter Stemming and Lemmatizing(둘 다 `nltk`에서 할 수 있다)은 "messages", "message", "messaging"등의 단어들을 한 단어로 다룰 수 있게 해주는 매우 유용한 도구다. 하지만 간단함을 위해, 이 튜토리얼에서는 여기까지만 하도록 하자.

#### Putting it all together

이 코드들을 재사용할 수 있도록 함수로 합치자:
    
    
    def review_to_words( raw_review ):
        # Function to convert a raw review to a string of words
        # The input is a single string (a raw movie review), and
        # the output is a single string (a preprocessed movie review)
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(raw_review).get_text()
        #
        # 2. Remove non-letters
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)
        #
        # 3. Convert to lower case, split into individual words
        words = letters_only.lower().split()
        #
        # 4. In Python, searching a set is much faster than searching
        #   a list, so convert the stop words to a set
        stops = set(stopwords.words("english"))
        #
        # 5. Remove stop words
        meaningful_words = [w for w in words if not w in stops]
        #
        # 6. Join the words back into one string separated by space,
        # and return the result.
        return( " ".join( meaningful_words ))

함수로 합치면서 두 가지가 달라졌는데, 첫번째는 속도를 위해 사용하는 `set`이고, 두번째는 최종적으로 추출한 단어들을 `join`으로 한 문장으로 합쳐 리턴한다.
    
    
    clean_review = review_to_words( train["review"][0] )
    print(clean_review)

이렇게 써 보면 동일한 결과를 확인할 수 있다. 이제, 모든 데이터를 클리닝 해 보자. 데이터가 25,000개나 되기 때문에 시간이 좀 걸린다:
    
    
    # Get the number of reviews based on the dataframe column size
    num_reviews = train["review"].size
    
    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []
    
    # Loop over each review; create an index i that goes from 0 to the length of the movie review list
    for i in range(num_reviews):
        if (i+1) % 1000 == 0:
            print("Review {0} of {1}".format(i+1, num_reviews))
        # Call our function for each one, and add the result to the list of clean reviews
        clean_train_reviews.append( review_to_words( train["review"][i] ) )

### Creating Features from a Bag of Words (Using scikit-learn)

자, 그럼 이제 우리에겐 깔끔하게 정리된 리뷰들이 있다. 그럼 이걸 어떻게 숫자로 표현(numeric representation)할 것인가? 수치화 시켜야 데이터 마이닝 알고리즘을 적용할 수 있다. 가장 기본적인 방법은 [Bag of Words](http://en.wikipedia.org/wiki/Bag-of-words_model)다. 이 방법은 모든 도큐먼트의 단어들을 모아서 bag 벡터(원문에서는 vocabulary라고 표현한다)를 만들고, 각 도큐먼트의 단어 등장 횟수를 세어 bag 벡터로 표현한다.

예를 들어 "hello world", "hello city"가 있다면, bag은 [hello, world, city]로 구성되고 따라서 위 두 도큐먼트는 각각 [1, 1, 0], [1, 0, 1]로 표현할 수 있다.

IMDB 데이터에는 수많은 리뷰가 있고 이는 커다란 bag을 형성한다. 이를 제한하기 위해 최대 bag의 크기를 정해야 한다. 여기서는, 5000개의 빈발 단어를 사용하기로 하자.

bag-of-words 특성을 추출하기 위해 **scikit-learn**의 `feature_extraction` 모듈을 사용한다. 
    
    
    print("Creating the bag of words...")
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000)
    
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    
    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()
    print(train_data_features.shape)

이후에 train_data_features.shape을 찍어 보면 (25000, 5000)의 매트릭스임을 확인할 수 있다. 위에서 볼 수 있듯이, `CountVectorzier`는 preprocessing, tokenization, stop word removal 등의 옵션을 제공한다. 자세한 건 [function documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)에서 확인하자. 이 튜토리얼에서는 과정을 단계별로 보여주기 위해 직접 구현하였다.

이제, Bag of Words 모델을 학습했으니 bag의 구성요소를 살펴보자.
    
    
    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    print(vocab)

원한다면, 각 단어들이 얼마나 등장했는지도 세어 볼 수 있다:
    
    
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)
    
    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    for tag, count in zip(vocab, dist):
        print(count, tag)

### Random Forest

자, 이제 우리에겐 Bag of Words로 수치화된 특성들과 각 특성에 해당하는 오리지널 감정 라벨들이 있다. 그럼 이제 supervised learning을 해 보자! 여기서는 scikit-learn에서 제공하는 [Random Forest classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)를 사용하고, 트리의 개수는 적절한 기본값인 100개로 설정한다. 트리의 개수를 늘리면 더 정확한 결과가 나오겠으나 더 오래 걸린다. 특성의 수도 마찬가지다. 참고로 100개도 충분히 오래 걸린다. 몇 분 걸릴 수 있으니 기다리자.
    
    
    print("Training the random forest...")
    from sklearn.ensemble import RandomForestClassifier
    
    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)
    
    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, train["sentiment"] )

### Creating a Submission

이제 남은 것은 학습된 Random Forest에 테스트셋을 넣어 결과를 뽑고, 이를 submission file로 출력하는 일이다. **testData.tsv**를 보면 25,000개의 리뷰와 아이디가 있다. 각각의 감정 라벨을 예측해야 한다.

아래에서 Bag of Words를 테스트셋에 적용할 때는 트레이닝셋에 사용했던 "fit_transform"이 아니라 "transform"을 사용한다는 것을 알아두자. "fit_transform"을 사용하면 우리의 모델이 테스트셋에 fit하게 되고, 다시 말해 테스트셋에 overfitting하게 된다. 이러한 이유로 테스트셋은 prediction을 하기 전에는 사용하지 않는다.

> 코드를 잘 보면, vectorizer가 fit_transform을 사용한다. 일반적으로 overfitting이라 함은 모델이 데이터에 overfitting하는 것이고, vectorzier는 단순히 데이터를 수치화 하는 것인데? 라는 의문이 들 수 있다. 
> 
> 그러나 다시 한번 생각해 보면, Bag of Words또한 데이터로부터 "학습" 하는 것이다. 즉, 이 경우에 테스트셋에 대해 fit_transform을 사용하게 되면 Random Forest classifier는 그대로지만 Bag of Words모델이 테스트셋에 오버피팅하게 되는 것이다. 결국, 테스트셋을 수치화 할 때에도 트레이닝 데이터로 만든 bag 벡터(vocabulary)를 기반으로 해야 한다는 것을 알 수 있다.
    
    
    # Read the test data
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t",quoting=3 )
    
    # Verify that there are 25,000 rows and 2 columns
    print(test.shape)
    
    # Create an empty list and append the clean reviews one by one
    num_reviews = len(test["review"])
    clean_test_reviews = []
    
    print("Cleaning and parsing the test set movie reviews...\n")
    for i in range(num_reviews):
        if( (i+1) % 1000 == 0 ):
            print("Review {0} of {1}".format(i+1, num_reviews))
        clean_review = review_to_words( test["review"][i] )
        clean_test_reviews.append( clean_review )
    
    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    
    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)
    
    # Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    
    # Use pandas to write the comma-separated output file
    output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

이제 드디어 submission을 할 수 있다! 여러가지를 수정해보고 결과를 비교해보자. 리뷰 클리닝을 다르게 해 보고, Bag of Words의 단어 수를 다르게 해 보고, Porter Stemming을 써 보고, 다른 classifier를 써 보는 등 다양한 걸 해 보자. 다른 데이터셋을 다뤄 보고 싶으면, [Rotten Tomatoes competition](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)에 도전해 보자. 또는, 완전히 다른 것에 대한 준비가 되었다면 Deep Learning and Word Vector 페이지로 가자!


[Tistory 원문보기](http://khanrc.tistory.com/99)
