# 텐서플로2와 머신러닝으로 시작하는 자연어처리

## 3장. 자연어처리 개요

### 단어 표현

### 텍스트 분류

### 텍스트 유사도

* 텍스트 유사도: 텍스트가 얼마나 유사한지를 표현하는 방식

  * 이 책에서는 딥러닝 기반의 텍스트 유사도 측정 방식 주로 사용

    * 단어, 형태소, 유사도 종류에 상관 없이 텍스트를 벡터화 한 후 벡터화 된 각 문장 간의 유사도를 측정

    * 단어 벡터화는 TF-IDF 이용

      * 벡터화 없이 유사도 측정 가능한 자카드 유사도 제외하고는 이 벡터화 값 사용

      ```python
      from sklearn.feature_extraction.text import TfidfVectorizer
      
      # 문장 벡터화 진행
      sent = ("문장1", "문장2")
      tfidf_vectorizer = TfidfVectorizer()
      tfidf_matrix = tfidf_vectorizer.fit_transform(sent)
      
      idf = tfidf_vectorizer.idf_
      print(dict(zip(tfidf_vectorizer.get_feature_names(), idf)))
      ```

  * 방법

    * 자카드 유사도(Jaccard Similarity)

      * 두 문장을 각각 단어의 집합으로 만든 뒤 두 집합을 통해 유사도 측정

      * 유사도 측정 방법

        * 공통된 단어의 개수를 전체 단어 수로 나누기

          * 0 ~ 1사이의 값, 1에 가까울 수록 높은 유사도

            ```python
            from sklearn.metrics import jaccard_score
            
            jaccard_score(np.array([1,1,0,0]), np.array([1,1,0,2]), average=None)
            ```

            

    * 코사인 유사도

      * 두 개의 벡터값에서 코사인 각도 구하는 방법

      * -1 ~ 1사이의 값, 1에 가까울수록 높은 유사도

      * 단순히 좌표상의 거리를 구하는 것이 아니라 두 벡터 간 각도를 구하는 방식이기 때문에 방향성의 개념이 더해져 다른 기법들보다 성능이 좋다.

        * 두 문장이 유사하다면 같은 방향 / 유사하지 않을수록 직교로 표현

          ```python
          from sklearn.metrics.pairwise import cosine_similarity
          
          cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
          ```

          

    * 유클리디언 유사도

      * 가장 기본적인 거리를 측정하는 공식

        * 여기서 구하는 거리는 유클리디언 거리(Euclidean Distance), L2 거리

      * n차원 공간에서 두 점 사이의 최단 거리를 구하는 방법

        ```python
        from sklearn.metrics.pairwise import euclidean_distances
        
        euclidean_distances(tfidf_matrix[0:1], tfidf_matrix[1:2])
        ```

      * 유클리디언 거리는 제한이 없기 때문에 표준화 시켜 특정 범위 내에 값이 들어올 수 있도록 만들어준다.

        * L1 정규화 : 각 벡터 안의 요소 값을 모두 더한 것의 크기가 1이 되도록 벡터 크기 조절하는 방법

          * 벡터 모든 값을 더한 뒤 이 값으로 각 벡터 값 나누기

            ```python
            import numpy as np
            
            def l1_normalize(v):
              norm = np.sum(v)
              return v / norm
            
            tfidf_norm_l1 = l1_normalize(tfidf_matrix)
            euclidean_distance(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2])
            ```

            

    * 맨하탄 유사도

      * 맨하탄 거리: 사각형 격자로 이뤄진 지도에서 출발점에서 도착점까지를 가로지르지 않고 갈 수 있는 최단거리 구하는 공식

        * 유클리디언 길이: L2 거리 / 맨하탄 거리:L1 거리

        * 마찬가지로 거리이기 때문에 표준화 필요

          ```python
          from sklearn.metrics.pairwise import manhattan_distance
          
          manhattan_distance(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2])
          ```

          

### 자연어 생성

### 기계 이해

* 기계 이해(Machine Comprehension): 기계가 어떤 텍스트에 대한 정보를 학습하고 사용자가 질의를 던졌을 때 그에 대해 응답하는 문제
* QA 태스크 데이터셋
  * bAbI: FacebookAI에서 기계가 데이터를 통해 합습해서 텍스트를 이해하고 추론하는 목적에서 만들어 진 데이터셋
  * SQuAD(Stanford Question Answering Dataset): 위키피디아에 있는 내용을 크라우드 소싱 해 만든 QA 데이터셋
    * EM(Exact Matching): 정답 위치와 완벽하게 일치하는지
    * F1: 정답 위치와 겹치는지
  * Visual QA: 이미지에 대한 정보와 텍스트 질의를 통해 이미지 컨텍스트에 해당하는 응답 알려주는 태스크



### 데이터 이해하기

* 탐색적 데이터 분석(EDA; Exploratory Data Analysis): 데이터를 이해하기 위한 방법으로 데이터의 패턴이나 잠재적 문제점을 발견할 수 있다.
  * 정해진 틀 없이 데이터에 대해 최대한 많은 정보를 뽑아내면 된다.
    * 데이터 평균, 중앙, 최소, 최대, 범위, 분포, 이상치 ...
    * 다양한 방법으로 시각화하며 데이터에 대한 직관 얻어야 한다.
  * 실습: 영화 리뷰 데이터(코드)
    * 파일 다루기
      * os.listdir(directory): 디렉토리 하위의 데이터 모두 가져오기(보통 for문과 함께 쓰는듯)
      * os.path.dirname(data_set): 디렉토리명 가져오기
      * os.path.join(directory, file_path): 디렉토리 + 하위디렉토리 합치기