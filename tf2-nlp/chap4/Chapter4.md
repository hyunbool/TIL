# 4장. 텍스트 분류

* 텍스트 분류: 자연어 처리 기술을 활용해 글의 정보를 추출 해 문제에 맞게 사람이 정한 범주(class)로 분류하는 문제
  * 이 장에서는 감정 분류 문제를 다룸



## 4.1 영어 텍스트 분류

* Bag of Words Meets Bag of Popcorn(워드 팝콘) 문제
  * https://www.kaggle.com/c/word2vec-nlp-tutorial
  * IMDB에서 나온 영화 평점 데이터를 활용한 캐글 문제
    * 각 데이터는 영화 리뷰 텍스트 & 평점에 따른 감정 값(긍/부정)으로 구성
  * 목표:
    * 데이터를 불러오는 것과 정제되지 않은 데이터를 활용하기 쉽게 전처리하는 과정
    * 데이터를 분석하는 과정
    * 문제 해결을 위해 알고리즘을 모델링 하는 과정

### 데이터 분석 및 전처리

* 과정

  1. 캐글 데이터 불러오기
  2. EDA
  3. 데이터 정제
     1. HTML 및 문장 부호 제거
     2. 불용어 제거
     3. 단어 최대 길이 설정
     4. 단어 패딩
     5. 벡터 표상화
  4. 모델링

* 데이터 불러오기

  * .zip 압축 풀기

    ```python
    zipref = zipfile.ZipFile('압축 풀 파일', 'r') // 객체 생성
    zipref.extractall('압축 풀 위치')
    zipref.close()
    ```

  *  히스토그램 작성

    ```python
    // 이미지 크기 선언
    plt.figure(figsize=(12, 5))
    
    // 히스토그램 선언
    plt.hist(train_length, bins='히스토그램 값에 대한 버킷 범위', alpha='그래프 색상 투명도', color='그래프 색상', label='그래프에 대한 라벨')
    plt.yscale('log', nonposy='clip')
    plt.title('그래프 제목')
    plt.xlabel('x축 라벨')
    plt.ylabel('y축 라벨')
    ```

  * 통계값 확인

    * np.max(): 최대값

    * np.min(): 최소값

    * np.mena(): 평균값

    * np.std(): 표준편차

    * np.median(): 중간값

    * np.percential(): 사분위(25: 1사분위, 75: 3사분위)

    * 박스플롯 확인

      ```python
      plt.boxplot(train_length, labels=['입력한 데이터에 대한 라벨'], showmenas='평균값 마크 여부')
      ```

      

  * 많이 사용된 단어 확인

    ``` python
    from wordcloud import WordCloud
    
    cloud = WordCloud(width, height).generate(" ".join(데이터))
    plt.figure(figsize=(,))
    plt.imshow(cloud)
    plt.axis('off')
    ```

    

  * 각 리뷰 당 단어 개수 확인

    * 각 단어의 길이를 가지는 변수 설정

      ```python
      train_word_counts = train_data['review'].apply(lambda x: len(x.split(' ')))
      ```

      

* 데이터 전처리

  * 라이브러리

    * 판다스: 데이터 다루기 위해
    * re, beautifulsoup: 데이터 정제 위해
    * NLTK의 stopwords 모듈: 불용어 제거
    * Tensorflow의 pad_sequence & Tokenizer
    * numpy: 데이터 저장

  * HTML 태그 & 특수문자 제거

    ```python
    from bs4 import BeautifulSoup
    review_text = BeautifulSoup(review, "html5lib").get_text() # html 태그 제거
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    ```

    

  * 불용어 삭제

    * NLTK의 불용어 사전은 소문자 단어로 구성되었기 때문에 불용어 제거를 위해서는 모든 단어를 소문자로 바꾼 후 불용어를 제거해야 한다.

      ```python
      from nltk.corpus import stopwords
      
      # 불용어 사전 만들기
      stop_words = set(stopwords.words('english'))
      
      review_text = review_text.lower()
      words = review_text.split()
      words = [w for w in words if not w in stop_words]
      ```

      

  * 단어 임베딩 생성

    ```python
    from tensorflow.python.keras.preprocessing.sequence import pad_sequences
    from tensorflow.python.keras.preprocessing.text import Tokenizer
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(clean_train_reviews)
    text_sequences = tokenizer.texts_to_sequences(clean_train_reviews)
    
    word_vocab = tokenizer.word_index # 단어 사전
    ```

  * 문장 길이 고정

    ```python
    MAX_LENGTH = 174 # 보통 중간값을 사용
    train_inputs = pad_sequences(text_sequences, maxlen=MAX_LENGTH, padding='post')
    ```

  * 전처리 데이터 저장



### 모델링

* 로지스틱 회귀 모델: 주로 이항 분류를 위해 사용되는 모델, 선형 결합을 통해 나온 결과를 토대로 예측

  * 선형 회귀 모델
    * 종속 변수와 독립변수 간 상관관계를 모델링 하는 방법
      * 하나의 선형 방정식으로 표현해 데이터를 분류하는 모델
    * 주어진 데이터 집합 <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/bb65235f66e69d8c663b673c5952ee7a64e9246d"/>에 대해 다음과 같은 선형 관계를 모델링하게 된다:

  <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/e8fe92790a76066af5556c62f5230bcc0bdf9f38">

  * 로지스틱 회귀 모델

    * 선형 모델의 결괏값에 로지스틱 함수를 적용해 0 ~ 1 사이의 값을 갖게 해서 확률로 표현

      * 결과에 1에 가까우면 정답이 1이라고 예측 / 0에 가까울 경우 0으로 예측

    * 두가지 임베딩 벡터 방법

      * TF-IDF: sklearn의 TfidVectorizer 사용

        * TF-IDF 벡터화

          ```python
          from sklearn.feature_extraction.text import TfidVectorizer
          
          vectorizer = TfidVectorizer(min_df = '설정한 값 보다 특정 토큰의 df값이 더 적게 나오면 벡터화 과정에서 제거', analyzer="분석 기준 단위", sublinear_tf='문서 빈도수에 대한 스무딩 여부', ngram_range=(n-gram 범위 설정), max_features='벡터의 최대 길이 설정')
          
          X = vectorizer.fit_transform(reviews)
          ```

        * 학습과 검증 데이터셋 분리

          ```python
          from sklearn.model_selection import train_test_split
          
          x_train, x_eval, y_train, y_eval = trian_test_split(x, y, test_size='학습/평가 데이터 비율')
          ```

        * 모델 선언 및 학습

          ```python
          from sklearn.linear_model import LogisticRegression
          
          lgs = LogisticRegression(class_weight='balanced') 
          lgs.fit(X_train, y_train) 
          ```

        * 성능 평가

          ```python
          print("Accuracy: %f" % lgs.score(X_eval, y_eval))
          ```

          

      * Word2Vec: 단어로 표현된 리스트를 입력값으로

        * word2vec 벡터화

          ```python
          num_features = 300 # 각 단어에 대해 임베딩 된 벡터 차원 정한다   
          min_word_count = 40 # 단어에 대한 최소 빈도 수
          num_workers = 4  # 학습을 위한 프로세스 개수
          context = 10   # word2vec 수행 위한 컨텍스트 윈도 크기
          downsampling = 1e-3 # 다운 샘플링 비율, 보통 0.01 사용
          ```

          ```python
          from gensim.models import word2vec
          
          model = word2vec.Word2Vec(sentences, workers=num_workers, \
                     size=num_features, min_count = min_word_count, \
                      window = context, sample = downsampling)
          ```

          

          * 문장 길이 고정

            * 문장 내 모든 단어의 벡터값에 대해 평균을 내 리뷰 하나 당 하나의 벡터로 만들기

              ```python
              def get_features(words, model, num_features):
                  # 출력 벡터 초기화
                  feature_vector = np.zeros((num_features),dtype=np.float32)
              
                  num_words = 0
                  
                  # 어휘 사전 준비
                  index2word_set = set(model.wv.index2word) 
              
                  # 해당 단어 임베딩 찾아 합 구하기
                  for w in words:
                      if w in index2word_set:
                          num_words += 1
                          feature_vector = np.add(feature_vector, model[w])
              
                  # 단어 전체 개수로 나눠 평균 벡터값 구하기
                  feature_vector = np.divide(feature_vector, num_words)
                  return feature_vector
              ```

* 랜덤 포레스트 분류 모델

  * 여러 개의 의사결정 트리의 결과값을 평균낸 것을 결과로 사용하는 머신러닝 모델

    * 의사 결정 트리: 자료구조 중 하나인 트리 구조와 같은 형태로 이뤄진 알고리즘
      * 각 노드는 하나의 질문이 되고, 질문에 따라 다음 노드가 달라진다.

  * CountVectorizer를 활용한 벡터화: 전처리 한 텍스트 데이터를 입력값으로 사용

    ``` python
    from sklearn.feature_extraction.text import CountVectorizer
    
    vectorizer = CountVectorizer(analyzer = "word", max_features = 5000) 
    
    train_data_features = vectorizer.fit_transform(reviews)
    ```

  * 모델 구현 및 학습

    ``` python
    from sklearn.ensemble import RandomForestClassifier
    
    
    # 랜덤 포레스트 분류기에  100개 의사 결정 트리를 사용한다.
    forest = RandomForestClassifier(n_estimators = 100) 
    
    # 단어 묶음을 벡터화한 데이터와 정답 데이터를 가지고 학습을 시작한다.
    forest.fit( train_input, train_label )
    ```

    

* 딥러닝 모델

  * 순환 신경망 분류 모델

    * 랜덤 시드 고정: 학습 상황 보존 위해

      ```python
      SEED_NUM = 1234
      tf.random.set_seed(SEED_NUM) # 랜덤 시드를 고정하면 학습을 하기 위한 랜덤 변수에 대한 초기 상태 유지 가능
      ```

    * 모델 하이퍼파라미터 정의

      ```python
      # 모델 학습을 위한 하이퍼파라미터
      model_name = 'rnn_classifier_en'
      BATCH_SIZE = 128
      NUM_EPOCHS = 5
      VALID_SPLIT = 0.1
      MAX_LEN = train_input.shape[1]
      
      # 모델 레이어의 차원 수 설정
      kargs = {'model_name': model_name,
              'vocab_size': prepro_configs['vocab_size'],
              'embedding_dimension': 100,
              'dropout_rate': 0.2,
              'lstm_dimension': 150,
              'dense_dimension': 150,
              'output_dimension':1}
      ```

    * 모델 구현

      ```python
      class RNNClassifier(tf.keras.Model): # 클래스로 모델 구현하기 위해 tf.keras.Model 상속
      	def __init__(self, **kargs):
      		# super 함수를 통해 부모 클래스에 있는 __init__ 함수 호출
      		# super 함수를 통해 부모 클래스의 __int__ 함수의 인자에 모델 이름을 전달하면, tf.keras.Model을 상속받은 모든 자식은 해당 모델의 이름으로 공통적으로 사용됨
      		super(RNNClassifier, self).__init__(name=kargs['model_name'])
      		self.embedding = layers.Embedding(input_dim=kargs['vocab_size'], output_dim=kargs['embedding_dimension'])
      		self.lstm_1_layer = tf.keras.layers.LSTM(kargs['lstm_dimension'], return_sequences=True)
      				self.lstm_2_layer = tf.keras.layers.LSTM(kargs['lstm_dimension'])
      				self.dropout = layers.Dropout(kargs['dropout_rate'])
      				#피드 포워드 네트워크 거치도록 한다.
      				self.fc1 = layers.Dense(units=kargs['dense_dimension'], activation=tf.keras.activations.tanh)
      				# 회귀 수행하는 레이어
      				self.fc2 = layers.Dense(units=kargs['output_dimension'], activation=tf.keras.activations.sigmoid)
      	
        # __init__으로 생성한 레이어는 call 함수를 통해 실행할 수 있다
      	def call(self, x):
      		x = self.embedding(x)
          x = self.dropout(x)
          x = self.lstm_1_layer(x)
          x = self.lstm_2_layer(x)
          x = self.dropout(x)
          x = self.fc1(x)
          x = self.dropout(x)
          x = self.fc2(x)
        
        	return x
      ```

    * 모델 생성

      ```python
      model = RNNClassifier(**kargs)
      model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])
      ```

    * 모델 학습

      * 오버피팅 현상 발생

      * 학습 도중 특정 상태의 모델에서 하이퍼파라미터 바꿔 다시 학습 진행

      * tensorflow.keras.callback 모듈의 EarlyStopping과 ModelCheckpoint 클래스 활용

        * EarlyStopping: 오버피팅 현상 방지하기 위해 특정 에폭에서 현재 검증 평가 점수가 이전 검증 평가 점수보다 일정 수치 미만으로 낮아지면 학습 멈추는 역할

        ```python
        earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=1)
        # min_delta: the threshold that triggers the termination (acc should at least improve 0.0001)
        # patience: no improvment epochs (patience = 1, 1번 이상 상승이 없으면 종료)
        ```
        * ModelCheckpoint: 에폭마다 모델을 저장하게 만들어 줌

          * save_best_only: 가장 성능이 좋은 모델만 저장

          * monitor: save_best_only의 평가 지표

          * save_weight_only: 모델 그래프 전부 저장하는 것이 아닌, 모델 가중치만 저장하는 옵션

            ``` python
            checkpoint_path = DATA_OUT_PATH + model_name + '/weights.h5'
            checkpoint_dir = os.path.dirname(checkpoint_path)
            
            # Create path if exists
            if os.path.exists(checkpoint_dir):
                print("{} -- Folder already exists \n".format(checkpoint_dir))
            else:
                os.makedirs(checkpoint_dir, exist_ok=True)
                print("{} -- Folder create complete \n".format(checkpoint_dir))
                
            
            cp_callback = ModelCheckpoint(
                checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
            ```

            

      * 모델 학습 시작

        ```python
        history = model.fit(train_input, train_label, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                            validation_split=VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])
        ```

        

  * 컨볼루션 신경망 분류 모델(CNN)

    * Yoon Kim(2014), *Convolutional Neural Network for Sentence Classification*

      * RNN: 단어 입력 순서 중요하게 반영 / CNN: 문장 지역 정보 보존하면서 각 문장의 성분 등장 정보를 학습에 반영하는 구조

      * 학습 시 각 필터 크기를 조절하며 언어의 특징 값 추출 -> n-gram 방식과 유사

        <img src="https://miro.medium.com/max/2800/0*0efgxnFIaLTZ2qkY"/>

    * 