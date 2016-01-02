---
layout: post
title: "[kaggle] Bag of Words Meet Bags of Popcorn - (3) Part 3: More Fun With Word Vectors"
tags: ['Text Mining']
date: 2015-07-22 15:27:00
---
# Part 3: More Fun With Word Vectors

## [Code](https://github.com/wendykan/DeepLearningMovies/blob/master/Word2Vec_BagOfCentroids.py)

## Numeric Representations of Words

이제 우리에겐 단어의 감정적 의미에 대해 이해하는 모델이 있다. 어떻게 써야 할까? 파트 2에서 학습된 Word2Vec 모델이 "syn0"이라는 `numpy` 배열에 저장된다. 
    
    
    >>> # Load the model that we created in Part 2
    >>> from gensim.models import Word2Vec
    >>> model = Word2Vec.load("300features_40minwords_10context")
    2014-08-03 14:50:15,126 : INFO : loading Word2Vec object from 300features_40min_word_count_10context
    2014-08-03 14:50:15,777 : INFO : setting ignored attribute syn0norm to None
    
    >>> type(model.syn0)
    <type 'numpy.ndarray'>
    
    >>> model.syn0.shape
    (16492, 300)

참고로, 모델을 트레이닝 할 때 myhashfxn을 사용했다면 로드하기 전에 동일한 해쉬함수를 정의해 놓아야 제대로 불러온다.

syn0의 row의 수 16,490은 파트 2에서 최소 word count를 40으로 설정한 것에 따른 모델의 vocabulary에 들어 있는 단어의 수이고, column의 수 300은 파트 2에서 설정한 특성(feature) 벡터의 크기다. 각 단어 벡터는 아래와 같이 살펴볼 수 있다:
    
    
    In[5]: model["flower"]

1x300 크기의 numpy array가 리턴된다.

## From Words To Paragraphs, Attempt 1: Vector Averaging

IMDB 데이터셋의 한 가지 문제는 리뷰의 길이가 변한다는 것이다. 우리는 여기서 각각의 단어 벡터들을 각 리뷰를 나타내는 동일한 크기의 특성 셋으로 나타내어야 한다.

모든 단어들이 전부 300차원 벡터이므로, 간단하게 각 리뷰의 벡터들을 평균 내는 방법을 사용할 수 있다 (이를 위해 stop word를 제거했다. 이러한 경우에 stop word는 노이즈가 된다). 

아래 코드들은 벡터들을 평균내는 함수다:
    
    
    import numpy as np  # Make sure that numpy is imported
    
    def makeFeatureVec(words, model, num_features):
        # Function to average all of the word vectors in a given
        # paragraph
        #
        # Pre-initialize an empty numpy array (for speed)
        featureVec = np.zeros((num_features,),dtype="float32")
        #
        nwords = 0.
        #
        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed
        index2word_set = set(model.index2word)
        #
        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its feature vector to the total
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec,model[word])
        #
        # Divide the result by the number of words to get the average
        featureVec = np.divide(featureVec,nwords)
        return featureVec
    
    
    def getAvgFeatureVecs(reviews, model, num_features):
        # Given a set of reviews (each one a list of words), calculate
        # the average feature vector for each one and return a 2D numpy array
        #
        # Initialize a counter
        counter = 0.
        #
        # Preallocate a 2D numpy array, for speed
        reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
        #
        # Loop through the reviews
        for review in reviews:
           #
           # Print a status message every 1000th review
           if counter%1000. == 0.:
               print("Review %d of %d" % (counter, len(reviews)))
           #
           # Call the function (defined above) that makes average feature vectors
           reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
           #
           # Increment the counter
           counter = counter + 1.
        return reviewFeatureVecs

이제 이 함수로 각 리뷰에 대한 평균 벡터를 구할 수 있다. 몇 분 걸릴 수 있다:
    
    
    import numpy as np  # Make sure that numpy is imported
    
    def makeFeatureVec(words, model, num_features):
        # Function to average all of the word vectors in a given
        # paragraph
        #
        # Pre-initialize an empty numpy array (for speed)
        featureVec = np.zeros((num_features,),dtype="float32")
        #
        nwords = 0.
        #
        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed
        index2word_set = set(model.index2word)
        #
        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its feature vector to the total
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec,model[word])
        #
        # Divide the result by the number of words to get the average
        featureVec = np.divide(featureVec,nwords)
        return featureVec
    
    def getAvgFeatureVecs(reviews, model, num_features):
        # Given a set of reviews (each one a list of words), calculate
        # the average feature vector for each one and return a 2D numpy array
        #
        # Initialize a counter
        counter = 0.
        #
        # Preallocate a 2D numpy array, for speed
        reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
        #
        # Loop through the reviews
        for review in reviews:
            #
            # Print a status message every 1000th review
            if counter%1000. == 0.:
                print("Review %d of %d" % (counter, len(reviews)))
            #
            # Call the function (defined above) that makes average feature vectors
            reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
            #
            # Increment the counter
            counter = counter + 1.
        return reviewFeatureVecs
    
    # ****************************************************************
    # Calculate average feature vectors for training and testing sets,
    # using the functions we defined above. Notice that we now use stop word
    # removal.
    
    clean_train_reviews = []
    for c, review in enumerate(train["review"]):
        if c%1000. == 0.:
            print("Training set {} of {}".format(c, train.shape[0]))
        clean_train_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
    
    trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )
    
    print("Creating average feature vecs for test reviews")
    clean_test_reviews = []
    for c, review in enumerate(test["review"]):
        if c%1000. == 0.:
            print("Test set {} of {}".format(c, test.shape[0]))
        clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
    
    testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

주석으로 설명이 적혀 있지만, 간단히 설명하자면 `makeFeatureVec`은 리뷰 파라그래프를 받아서 각 단어들에 대해 model이 포함하는 단어인지 검사하여 평균을 구한다. 즉, 특정 리뷰에 대해 특성 벡터를 구하는 함수이고 `getAvgFeatureVecs`는 모든 리뷰에 대해 `makeFeatureVec`함수를 적용하여 특성 벡터 리스트를 구하는 함수다.

자, 그럼 이제 각 리뷰들의 특성 벡터를 추출하였으니 이 값으로 머신러닝 알고리즘을 돌릴 수 있다. Bag of Words에서 했던 것처럼 랜덤 포레스트를 적용해 보자.
    
    
    # Fit a random forest to the training data, using 100 trees
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier( n_estimators = 100 )
    
    print("Fitting a random forest to labeled training data...")
    forest = forest.fit( trainDataVecs, train["sentiment"] )
    
    # Test & extract results
    result = forest.predict( testDataVecs )
    
    # Write the test results
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )

이제 이 결과를 제출하면 얼마나 잘 예측했는지를 볼 수 있는데, 오히려 Bag of Words보다 결과가 안 좋다!

원소별로 평균을 내는 방법이 썩 좋은 결과를 보이지 못했다. 어떻게 이를 개선할 수 있을까? 일반적인 방법은 [tf-dif](http://en.wikipedia.org/wiki/Tf%E2%80%93idf)를 사용해서 단어 벡터에 가중치를 부여하는 방법이다. scikit-learn에서 제공하는 [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)를 사용해서 간단하게 구현할 수 있다. 그런데 실제로 적용해 보았을 때 별다른 성능 향상이 없었다.

## From Words to Paragraphs, Attempt 2: Clustering

Word2Vec은 의미가 유사한 단어들의 클러스터를 만든다. 이를 이용해서 클러스터에서 단어 유사도를 살펴보는 접근방법을 적용해 보자. 이렇게 벡터들을 그루핑(grouping) 하는 방법을 "vector quantization" 이라고 한다. 이를 위해 [K-Means](http://en.wikipedia.org/wiki/K-means_clustering) 클러스터링 알고리즘을 사용한다.

K-Means 알고리즘에서는 클러스터의 수 "K"를 설정해 주어야 하는데, 이를 어떻게 정할까? 여러 K를 시도해 본 결과, 평균적으로 클러스터당 5개 단어 정도의 작은 클러스터가 적합했다. 작은 클러스터를 사용한다는 것은 반대로 클러스터의 수, 즉 K가 굉장히 크다는 것이고 이는 오랜 트레이닝 시간을 필요로 한다. 원문에서 저자의 컴퓨터에서는 40분 이상이 걸렸다고 하는데, 내 컴퓨터에서는 11분(675 초) 정도 걸렸다.
    
    
    from sklearn.cluster import KMeans
    import time
    
    start = time.time() # Start time
    
    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.syn0
    num_clusters = int(word_vectors.shape[0] / 5)
    
    print("the number of clusters: {}".format(num_clusters))
    
    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans( n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( word_vectors )
    
    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print("Time taken for K Means clustering: ", elapsed, "seconds.")

여러 가지로 테스트 해 본 결과, 500개의 단어를 하나의 클러스터로 하면 32개의 클러스터가 나오며 30초 가량 걸린다. 클러스터당 50개의 단어로 하면 329개의 클러스터가 나오고 90초 정도 걸린다. 클러스터당 5개의 단어인 경우에는 3298개의 클러스터가 나오고, 11분이 걸린다.

클러스터링이 끝나면, 각 클러스터에 포함된 단어의 인덱스들이 `idx`배열에 저장된다. 이를 `model.index2word`와 묶어서 단어와 매핑하자.
    
    
    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number                                                                                            
    word_centroid_map = dict(zip( model.index2word, idx ))

word_centroid_map에는 단어가 어떤 클러스터 소속인지 저장된다.
    
    
    In[19]: word_centroid_map
    Out[19]: 
    {'toys': 444,
     'overly': 401,
     'devil': 1079,
     'rightful': 2008,
     'suburban': 719,
     ...

클러스터 별로 조금 더 자세히 살펴보자.
    
    
    # For the first 10 clusters
    for cluster in range(0, 10):
        #
        # Print the cluster number
        print("\nCluster {}".format(cluster))
        #
        # Find all of the words for that cluster number, and print them out
        words = []
    
        for k, v in word_centroid_map.items():
            if v == cluster:
                words.append(k)
        print(words)

이 코드를 돌려보면, 아래와 같은 결과를 얻을 수 있다.
    
    
    Cluster 0
    ['noble', 'brave']
    
    Cluster 1
    ['unworthy', 'pretentiousness', 'ineptness', 'notwithstanding', 'overwhelm']
    
    Cluster 2
    ['tomas', 'milian', 'nero', 'franco', 'jess']
    
    Cluster 3
    ['poker', 'picnic', 'carnival', 'golf', 'pond', 'cafe', 'digs', 'parlor', 'cane']
    
    Cluster 4
    ['companionship', 'friendships', 'individuality', 'frailty', 'passions', 'harmony', 'inspires', 'pleasures', 'elusive']
    
    Cluster 5
    ['edna']
    
    Cluster 6
    ['straight', 'direct']
    
    Cluster 7
    ['crafty', 'manic']
    
    Cluster 8
    ['matched', 'impeccable', 'vocal', 'matching', 'sparkling']
    
    Cluster 9
    ['beats', 'messes', 'cooks', 'nuts', 'lighten']

살펴보면 클러스터마다 퀄리티가 다양하다. 비슷한 단어끼리 묶인 클러스터가 있는가 하면, 쌩뚱맞은 조합도 존재한다. 원문의 결과와는 완전히 다른데, 이는 word_centroid_map이 dictionary라서 10개를 뽑으면 랜덤하게 뽑히기 때문에 그렇다.

다음 단계로 넘어가기 전에, K-Means 클러스터링에 너무 오랜 시간이 걸리므로 word_centroid_map을 피클링하자.
    
    
    # 자 그럼 이제 word_centroid_map을 피클링하자.
    print("word_centroid_map dumping ...")
    with open("word_centroid_map.pickle", "wb") as f:
        import pickle
        pickle.dump(word_centroid_map, f)

이제 클러스터링을 새로 하지 않고 저장된 파일로부터 word_centroid_map을 불러올 수 있다.
    
    
    # load word_centroid_map
    print("word_centroid_map loading ...")
    with open("word_centroid_map.pickle", "rb") as f:
        import pickle
        word_centroid_map = pickle.load(f)

자 이제 클러스터간 퀄리티가 왔다갔다하긴 하지만, 클러스터링된 단어들, 바꿔 말하면 각각 centroid를 갖고 있는 단어들을 확보했다. 이제 이를 사용해서 bags-of-centroids를 만들 수 있다!

> 결국 비슷한 단어들끼리 묶어서 Bag of Words를 하는 것이다. 비슷한 단어를 묶었으니 Bag of Clusters가 되는 셈이고, cluster가 곧 centroid이니 Bag of Centroids이다. 같은 단어의 형변환을 묶어주는 Stemming이나 Lemmatizing에서 한 단계 더 나아간 형태라고 볼 수 있다.
    
    
    def create_bag_of_centroids( wordlist, word_centroid_map ):
        #
        # The number of clusters is equal to the highest cluster index
        # in the word / centroid map
        num_centroids = max( word_centroid_map.values() ) + 1
        #
        # Pre-allocate the bag of centroids vector (for speed)
        bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
        #
        # Loop over the words in the review. If the word is in the vocabulary,
        # find which cluster it belongs to, and increment that cluster count
        # by one
        for word in wordlist:
            if word in word_centroid_map:
                index = word_centroid_map[word]
                bag_of_centroids[index] += 1
        #
        # Return the "bag of centroids"
        return bag_of_centroids

Bag of Words와 유사하게, Bag of Centroids를 계산하는 함수다. 마찬가지로 이 함수를 아까 단어 리스트로 정제한 리뷰들에 적용해서 우리의 데이터셋에 대한 Bag of Centroids를 만들자.
    
    
    # Pre-allocate an array for the training set bags of centroids (for speed)
    train_centroids = np.zeros( (train["review"].size, num_clusters), dtype="float32" )
    
    # Transform the training set reviews into bags of centroids
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
        counter += 1
    
    # Repeat for test reviews
    test_centroids = np.zeros(( test["review"].size, num_clusters), dtype="float32" )
    
    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
        counter += 1

이렇게 만든 Bag of Centroids를 사용해서 다시 랜덤 포레스트를 돌려보자.
    
    
    # Fit a random forest and extract predictions
    forest = RandomForestClassifier(n_estimators = 100)
    
    # Fitting the forest may take a few minutes
    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(train_centroids,train["sentiment"])
    result = forest.predict(test_centroids)
    
    # Write the test results
    output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
    output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )

이렇게 돌리면 파트 1의 Bag of Words와 비슷하거나 살짝 안좋은 결과를 보여준다.


[Tistory 원문보기](http://khanrc.tistory.com/110)
