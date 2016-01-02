---
layout: post
title: "Text Mining Tutoral: 20 newsgroups"
tags: ['Text Mining']
date: 2015-06-02 11:02:00
---
# Text Mining Tutoral: 20 newsgroups

`scikit-learn`의 텍스트 데이터를 다루는 튜토리얼이다.  
이 튜토리얼에서 우리는 이러한 것들을 다룰 것이다:

  * 파일 내용과 카테고리를 로드하기
  * 머신러닝에 알맞은 특성 벡터를 추출하기
  * 분류하기 위해 리니어 모델을 학습하기
  * 특성 추출과 분류기 학습에 대해 좋은 파라메터를 찾기 위해 그리드 서치(grid search) 사용하기

요약이니 자세한 내용은 [원문](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#tokenizing-text-with-scikit-learn)을 참고하자.

## Tutorial setup

여기서는 Window/Anaconda 환경에서 해 보았다. `sklearn`만 잘 깔려있으면 문제없을 것이다. 원문에서 말하는 데이터셋은 찾지 못했지만 필요하지 않다.

## Loading the 20 newsgroups dataset

이 데이터셋은 "Twenty Newsgroups"라고 불린다. 20개의 뉴스그룹으로 분리된 20,000여개의 뉴스그룹 도큐먼트다. 이 데이터는 머신러닝에서의 대표적인 튜토리얼용 텍스트 데이터이다.
    
    
    from sklearn.datasets import fetch_20newsgroups
    
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    

이렇게 데이터를 불러올 수 있다. `sklearn`에서 데이터를 다운받고 쉽게 원하는 부분만 사용할 수 있도록 제공한다.

## Extracting features from text files

텍스트 마이닝을 위해, 먼저 텍스트를 numerical feature vector로 바꿀 필요가 있다.

### Bags of words

가장 직관적인 방법이다.  
X[i, j] = #(w)  
도큐먼트 i에서 단어 w가 등장하는 횟수. j는 단어 w의 인덱스.

보통 단어는 100,000개 이상인데, 샘플의 수가 10000개라고 한다면, float32로 구성된 numpy array에 이를 저장한다고 할 때 10000 x 100000 x 4 byte = **4GB in RAM** 이 필요하다. 다행히도 X의 대부분의 값은 0이고, 따라서 bags of words는 **high-dimensional sparse datasets**이다. 이러한 데이터는 0이 아닌 값만 저장하여 메모리를 아낄 수 있다.

`scipy`에서는 **scipy.sparse** 라는 여기에 딱 맞는 자료구조를 제공한다.

### Tokenizing text with scikit-learn

특성 딕셔너리를 만들고 도큐먼트를 특성 벡터로 변환하기 위해, stopwords의 텍스트 프리프로세싱, 필터링 그리고 토크나이징이 지원된다.
    
    
    from sklearn.feature_extraction.text import CountVectorizer
    
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    print(X_train_counts.shape)
    // (2257, 35788)
    

도큐먼트의 수(데이터 샘플 수)가 총 2257개 이고, 35788은 아마 총 word의 수 일 것 같다.

또한 CountVectorizer는 단어의 카운팅도 지원한다. vocabulary가 dictionary형태로 들고 있다.
    
    
    print(count_vect.vocabulary_.get('algorithm'))
    print(count_vect.vocabulary_['algorithm'])
    print(count_vect.__class__)
    print(count_vect.vocabulary_.__class__)
    // 4690
    // 4690
    // <class 'sklearn.feature_extraction.text.CountVectorizer'>
    // <class 'dict'>
    

### From occurrences to frequencies

Occurence는 나쁘지 않은 시작이지만 문제가 많다. [TF-IDF](http://en.wikipedia.org/wiki/Tf%E2%80%93idf)(Term Frequency times Inverse Document Frequency)가 많이 쓰인다. 먼저 **TF**를 구해보자.
    
    
    from sklearn.feature_extraction.text import TfidfTransformer
    
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    

위 코드에서 `fit()`함수와 `transform()`함수는 `fit_transform()`함수로 통합하여 중복 계산을 줄일 수 있다. 이 방법을 사용해서 **TF-IDF**를 구해보자.
    
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    

각각의 shape를 찍어보고, [0]에 들어있는 내용을 확인해 보면 차이를 알 수 있다.
    
    
    print(X_train_counts.shape)
    print(X_train_tf.shape)
    print(X_train_tfidf.shape)
    
    print(X_train_counts[0])
    print(X_train_tf[0])
    print(X_train_tfidf[0])
    

## Training a classifier

이제 우리의 특성을 갖게 되었으니, 분류기(classifier)를 학습할 수 있다. [_Naive Bayes_](http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes)를 학습해 보자. `sklearn`은 다양한 NB(Naive Bayes)를 학습할 수 있다. word count에 가장 적합한 건 multinomial 버전이다.
    
    
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
    

자 그럼 이제 새로운 도큐먼트를 데이터로 삼아 predict를 해 봐야 할 텐데, 그러기 위해 이전에 특성 추출을 한 것과 같이 새로운 도큐먼트에 대해서도 특성 추출이 필요하다. 위에서 했던 과정과의 차이는, 이미 training data에 대해 fit해 있기 때문에 `fit_transform()`이 아닌 `transform()`을 사용한다.

> min-max 노멀라이징을 할 때, training data에 대해 min-max노멀라이징을 한 뒤 해당 min-max를 기억하고 있다가 new data에 대해서도 동일한 min-max를 적용하는 데, 이와 같다. fit이라는 게 이런 의미인 듯함.
    
    
    docs_new = ['God is love', 'OpenGL on the GPU is fast']
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    predicted = clf.predict(X_new_tfidf)
    
    for doc, category in zip(docs_new, predicted):
        print("{} => {}".format(doc, twenty_train.target_names[category]))
    
    // God is love => soc.religion.christian
    // OpenGL on the GPU is fast => comp.graphics
    

여기까지 하면 분류기의 생성까지 완료된 것인데, 지금까지의 작업 수행 시간을 살펴보면 대략 총 1초가 걸린다:
    
    
    [Elapsed time] get data: 0.24381279945373535
    [Elapsed time] get bags of words(vectorize): 0.725600004196167
    [Elapsed time] get TF & TF-IDF: 0.02526402473449707
    [Elapsed time] learn multinomial NB: 0.010046005249023438
    

## Building a pipeline

지금까지 한 작업을 정리해 보면, vectorizer =&gt; transformer =&gt; classifier 의 세 단계다. 이 작업을 더 쉽게 할 수 있도록 `sklearn`은 `Pipeline`을 제공한다.
    
    
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])
    text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
    
    // predicted = text_clf.predict(docs_new) // differenct!
    

이렇게 `pipeline`으로 묶으면 수행시간도 0.7초 정도로 30%가량 빨라진다. 위의 'vect', 'tfidf', 'clf'는 마음대로 지정하면 되는데 이후 `grid search`에서 사용한다.

> 여기서 최종적으로 얻은 text_clf는 위에서 얻은 clf와 조금 다른데, predict 과정에서 사용하는 데이터가 TF-IDF 데이터가 아닌 그냥 도큐먼트다.

## Evaluation of the performance on the test set
    
    
    import numpy as np
    
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    start = time()
    docs_test = twenty_test.data
    predicted = text_clf.predict(docs_test)
    
    print(len(docs_test))
    print(np.mean(predicted == twenty_test.target))
    
    // 1502
    // 0.834886817577
    

참고로 1502개의 테스트 데이터를 가져오는 데에는 0.2초 가량이, predict하는 데에는 0.45초 가량이 소요되었다.

이렇게 `Naive Bayes`를 통해서 83.4%의 정확도를 확보하였다. 이번엔 [_Support Vector Machine (SVM)_](http://scikit-learn.org/stable/modules/svm.html#svm)을 사용해서 정확도를 개선해 보자. `SVM`은 가장 널리 알려진 강력한 텍스트 분류 알고리즘이다. 물론 `Naive Bayes`에 비하면 조금 느리지만. 우리는 파이프라인에서 classifier를 바꿔 끼우는 것만으로 학습기를 변경할 수 있다.
    
    
    from sklearn.linear_model import SGDClassifier
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
    _ = text_clf.fit(twenty_train.data, twenty_train.target)
    
    predicted = text_clf.predict(docs_test)
    print(np.mean(predicted == twenty_test.target))
    
    // 0.912782956059
    

분류기의 학습 시간은 0.75초 가량으로 0.05초 정도가 더 소요되었고, 분류기를 사용하여 예측하는 시간은 0.45초 정도로 `Naive Bayes`와 비슷했다.

단순히 정확도를 보는 것 외에, `sklearn`은 더 자세하게 결과를 분석할 수 있도록 해 준다:
    
    
    from sklearn import metrics
    print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
    print(metrics.confusion_matrix(twenty_test.target, predicted))
    
    // report
                            precision    recall  f1-score   support
    
               alt.atheism       0.94      0.82      0.87       319
             comp.graphics       0.88      0.98      0.92       389
                   sci.med       0.95      0.89      0.92       396
    soc.religion.christian       0.90      0.95      0.92       398
    
               avg / total       0.92      0.91      0.91      1502
    
    // confusion matrix
    [[261  10  12  36]
     [  5 380   2   2]
     [  7  32 353   4]
     [  6  11   4 377]]
    

예상대로 confusion matrix는 atheism/christian 뉴스그룹간에 혼동이 많이 일어난다.

> 결과가 원문과 살짝 다른데 이유는 잘 모르겠음.

## Parameter tuning using grid search

이미 여러번 등장했지만 분류기의 학습에는 수많은 파라메터들이 필요하다. 이 파라메터의 확인은 모듈의 도큐먼트를 확인하거나 파이썬의 `help`함수를 사용하자.

`Grid Search`를 이용하면, 이러한 파라메터들에 대한 테스트를 손쉽게 진행할 수 있다. SVM에서 단어별로 구분할 것인지 bigram(2글자)으로 구분할 것인지, idf를 쓸지 말지, penalty 파라메터의 값을 0.01로 할지 또는 0.001로 할지.
    
    
    from sklearn.grid_search import GridSearchCV
    
    parameters = {'vect__ngram_range': [(1,1), (1,2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3)}
    

이와 같은 exhaustive search(전역검색)는 매우 비싼 작업이다. 만약 우리가 여러개의 CPU를 사용할 수 있다면 `n_jobs`파라메터를 사용해서 parallel하게 작업을 수행할 수 있다. 만약 이 값에 `-1`을 준다면, 그리드 서치가 알아서 코어의 수를 체크하고 모두 사용하여 작업을 수행한다.
    
    
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
    

위와 같이, grid search도 보통의 sklearn의 분류기 학습처럼 진행된다. 단, 데이터셋을 그대로 쓰면 너무 느릴 수 있으니 작은 데이터셋을 사용하자.

> (아마도)윈도우 환경에서는 그리드 서치를 하려면 `if __name__ == "__main__":` 이 필요하다. 없이 하면 에러가 난다.
    
    
    print(twenty_train.target_names[gs_clf.predict(['God is love'])])
    // soc.religion.christian
    
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    // clf__alpha: 0.001
    // tfidf__use_idf: True
    // vect__ngram_range: (1, 1)
    
    print(score)
    // 0.9025
    

결국 분류기가 학습되는 과정이므로 위와 같이 predict를 해볼 수 있다. 아마 가장 좋은 분류기가 들어있을 것으로 보인다. 또한 이때의 파라메터들을 확인해 볼 수 있다.

## Fianlly

원문을 보면, 추가로 더 학습할 수 있도록 다양한 예제들과, 이후에 뭘 해야 할 지 가이드라인을 제시한다. 더 공부하고자 하면 이를 참고해서 공부하도록 하자.


[Tistory 원문보기](http://khanrc.tistory.com/94)
