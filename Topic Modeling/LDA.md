# Topic Modeling

## LDA(Latent Dirichlet Allocation)

### 모델 개요

<img src="http://i.imgur.com/r5e5qvs.png"/>

* LDA: 주어진 문서에 대해 각 문서에 어떤 주제들이 존재하는지에 대한 확률 모형
  * 토픽별 단어의 분포, 문서별 토픽의 분포를 모두 측정
    * 특정 토픽에 특정 단어가 나타날 확률 계산
      * 말뭉치로부터 얻은 토픽 분포로부터 토픽 선정
      * 해당 토픽에 해당하는 단어 뽑기
      * => LDA가 가정하는 문서 생성 과정
  * LDA는 학습을 통해 어떤 단어가 어떤 토픽에서 뽑힌 단어인지에 대한 잠재 정보를 알아낸다.



### 모델 구조

* D: 말뭉치 전체 문서 개수 / K: 전체 토픽 수(하이퍼 파라미터) / N: d번째 문서의 단어 수
  * 네모 칸: 반복 / 동그라미: 변수

<img src="https://image.slidesharecdn.com/lda-190707065621/95/latent-dirichlet-allocation-6-638.jpg?cb=1562482607"/>

#### 변수

* <img src="https://latex.codecogs.com/gif.latex?%5Cphi_%7Bk%7D"/>: per-corpus topic distributions, k번째 토픽에 해당하는 벡터
  * 길이는 말뭉치 전체 단어 개수만큼
  * 벡터 내 각 요소들은 해당 단어가 k번째 토픽에서 차지하는 비중을 나타냄
  * 각 요소는 확률이므로 모든 요소의 합은 1이 된다!
  * 하이퍼 파라미터 β의 영향을 받음
    * LDA는 토픽의 단어 비중 <img src="https://latex.codecogs.com/gif.latex?%5Cphi_%7Bk%7D"/>이 디리클레분포를 따른다는 가정을 취하기 때문
* <img src="https://latex.codecogs.com/gif.latex?%5Ctheta_%7Bd%7D"/>: per-document topic proportions, d번째 문서가 가진 토픽 비중 나타내는 벡터
  * 전체 토픽 개수 K만큼의 길이를 가짐
  * 각 요소값은 k번째 토픽이 해당 d번째 문서에서 차지하는 비중을 나타냄
  * 마찬가지로 확률이므로 모든 요소의 합은 1이 된다.
  * 하이퍼 파라미터 α의 영향을 받음
* <img src="https://latex.codecogs.com/gif.latex?z_%7Bd%2C%20n%7D"/>: per-word topic assigment, d번째 문서의 n번째 단어가 어떤 토픽에 해당하는지 할당해주는 역할
* <img src="https://latex.codecogs.com/gif.latex?w_%7Bd%2C%20n%7D"/>: 문서에 등장하는 단어를 할당해주는 역할



#### inference

* <img src="https://latex.codecogs.com/gif.latex?w_%7Bd%2C%20n%7D"/>를 가지고 잠재 변수를 역으로 추정하는 inference 과정을 살펴보자.
  * LDA => 토픽의 단어 분포와 문서의 토픽 분호의 결합으로 문서 내 단어들이 생성된다고 가정
  * 실제 관찰가능한 문서 내 단어를가지고 우리가 알고 싶은 토픽의 단어 분포, 문서의 토픽 분포를 추정하는 과정
* 토픽의 단어 분포와 문서의 토픽분포의 결합 확률이 커지도록 만들어야 한다.

<img src="http://i.imgur.com/ArQyvuO.png"/>

<div align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20p%28%26%7B%20%5Cphi%20%7D_%7B%201%3AK%20%7D%2C%7B%20%5Ctheta%20%7D_%7B%201%3AD%20%7D%2C%7B%20z%20%7D_%7B%201%3AD%20%7D%2C%7B%20w%20%7D_%7B%201%3AD%20%7D%29%3D%5C%5C%20%26%5Cprod%20_%7B%20i%3D1%20%7D%5E%7B%20K%20%7D%7B%20p%28%7B%20%5Cphi%20%7D_%7B%20i%20%7D%7C%5Cbeta%20%29%20%7D%20%5Cprod%20_%7B%20d%3D1%20%7D%5E%7B%20D%20%7D%7B%20p%28%7B%20%5Ctheta%20%7D_%7B%20d%20%7D%7C%5Calpha%20%29%20%7D%20%5Cleft%5C%7B%20%5Cprod%20_%7B%20n%3D1%20%7D%5E%7B%20N%20%7D%7B%20p%28%7B%20z%20%7D_%7B%20d%2Cn%20%7D%7C%7B%20%5Ctheta%20%7D_%7B%20d%20%7D%29p%28w_%7B%20d%2Cn%20%7D%7C%7B%20%5Cphi%20%7D_%7B%201%3AK%20%7D%2C%7B%20z%20%7D_%7B%20d%2Cn%20%7D%29%20%7D%20%5Cright%5C%7D%20%5Cend%7Balign*%7D"/>
</div>





* 𝑝(𝑧, ϕ, θ|𝑤)를 최대로 만드는 𝑧, ϕ, θ를 찾아야 한다!
  * 사후 확률을 계산하기 위해서는 분모에 해당하는 𝑝(𝑤)를 구해야 한다.
    * 베이즈 정리에서 evidence라 불리는 것으로 잠재 변수 𝑧, ϕ, θ의 모든 경우의 수를 고려한 각 단어(w)의 등장 확률을 가리킨다.
    * 하지만 현실적으로  𝑧, ϕ, θ는 직접 관찰하는게 불가능하며, 𝑝(𝑤)를 구할 때  𝑧, ϕ, θ의 모든 경우를 고려해야 함.
      * **깁스 샘플링** 같은 기법을 사용한다.

#### LDA와 깁스 샘플링

* collapsed gibbs sampling: 나머지 변수는 고정시킨 채 한 변수만을 변화시키되 일부 변수는 샘플링에서 제외하는 기법

  <img src="https://latex.codecogs.com/gif.latex?p%28%7B%20z%20%7D_%7B%20i%20%7D%3Dj%7C%7B%20z%20%7D_%7B%20-i%20%7D%2Cw%29"/>

  * w와 <img src="https://latex.codecogs.com/gif.latex?%7Bz%20%7D_%7B%20-i%20%7D"/>가 주어졌을 때, 문서의 i번째 단어의 토픽이 j일 확률
  * w: 말뭉치가 주어졌기 때문에 이미 알고있는 값
  * z: 각 단어가 어떤 토픽에 할당되어 있는지를 나태내는 변수
    * <img src="https://latex.codecogs.com/gif.latex?%7Bz%20%7D_%7B%20-i%20%7D"/>: i번째 단어의 토픽 정보를 제외한 모든 단어의 토픽 정보

* 계산과정
  * d번째 문서의 i번째 단어의 토픽 <img src="https://latex.codecogs.com/gif.latex?z_%7Bd%2C%20i%7D"/>이 j번째 토픽에 할당 될 확률

<img src="https://latex.codecogs.com/gif.latex?p%28%7B%20z%20%7D_%7Bd%2C%20i%20%7D%3Dj%7C%7B%20z%20%7D_%7B%20-i%20%7D%2Cw%29%3D%5Cfrac%20%7B%20%7B%20n%20%7D_%7B%20d%2Ck%20%7D&plus;%7B%20%5Calpha%20%7D_%7B%20j%20%7D%20%7D%7B%20%5Csum%20_%7B%20i%3D1%20%7D%5E%7B%20K%20%7D%7B%20%28%7B%20n%20%7D_%7B%20d%2Ci%20%7D&plus;%7B%20%5Calpha%20%7D_%7B%20i%20%7D%29%20%7D%20%7D%20%5Ctimes%20%5Cfrac%20%7B%20%7B%20v%20%7D_%7B%20k%2C%7B%20w%20%7D_%7B%20d%2Cn%20%7D%20%7D&plus;%7B%20%5Cbeta%20%7D_%7B%20%7B%20w%20%7D_%7B%20d%2Cn%20%7D%20%7D%20%7D%7B%20%5Csum%20_%7B%20j%3D1%20%7D%5E%7B%20V%20%7D%7B%20%28%7B%20v%20%7D_%7B%20k%2Cj%20%7D&plus;%7B%20%5Cbeta%20%7D_%7B%20j%20%7D%29%20%7D%20%7D%3DAB"/>

<img src="img/1.png"/>



* 모든 문서, 단어에 대해 깁스 샘플링 수행하면 모든 단어마다 토픽을 할당할 수 있으며, 이 과정에서 ϕ, θ 도 구할 수 있다.



#### 디리클레 파라미터

* d번째 문서에 i번째로 등장하는 단어의 토픽이 j일 확률:

  <img src="https://latex.codecogs.com/gif.latex?p%28%7B%20z%20%7D_%7Bd%2C%20i%20%7D%3Dj%7C%7B%20z%20%7D_%7B%20-i%20%7D%2Cw%29%3D%5Cfrac%20%7B%20%7B%20n%20%7D_%7B%20d%2Ck%20%7D&plus;%7B%20%5Calpha%20%7D_%7B%20j%20%7D%20%7D%7B%20%5Csum%20_%7B%20i%3D1%20%7D%5E%7B%20K%20%7D%7B%20%28%7B%20n%20%7D_%7B%20d%2Ci%20%7D&plus;%7B%20%5Calpha%20%7D_%7B%20i%20%7D%29%20%7D%20%7D%20%5Ctimes%20%5Cfrac%20%7B%20%7B%20v%20%7D_%7B%20k%2C%7B%20w%20%7D_%7B%20d%2Cn%20%7D%20%7D&plus;%7B%20%5Cbeta%20%7D_%7B%20%7B%20w%20%7D_%7B%20d%2Cn%20%7D%20%7D%20%7D%7B%20%5Csum%20_%7B%20j%3D1%20%7D%5E%7B%20V%20%7D%7B%20%28%7B%20v%20%7D_%7B%20k%2Cj%20%7D&plus;%7B%20%5Cbeta%20%7D_%7B%20j%20%7D%29%20%7D%20%7D%3DAB"/>

  * A: d번째 문서가 j번째 토픽과 맺고 있는 연관성 정도 / B: d번째 문서의 n번째 단어(w_{d, n})가 j번째 토픽과 맺고 있는 연관성의 강도
  * 그렇다면 하이퍼파라미터 α, β의 역할은?

* 특정 토픽에 할당된 단어들이 없는 경우에도 값이 0으로 계산되는 것을 막는 smoothing 역할을 수행

  * α가 클수록 토픽들의 분포가 비슷해지고, 작을 수록 특정 토픽이 크게 나타나게 된다!

<img src="http://i.imgur.com/zgXrEKI.png"/>



#### 최적 토픽 수 찾기

* 토픽 수 K는 사용자가 지정하는 하이퍼파라미터

* LDA에서 문서가 생성되는 과정 => 확률모형

  * LDA로부터 추정된 토픽 정보(z)를 활용해 계산한 각 단어의 발생 확률이 클수록 학습 코퍼스가 생성되는 과정을 제대로 설명한 것이 된다.
  * 최적 토픽 수를 찾기 위한 방법도 이 아이디어를 사용

* 모든 문서와 단어의 발생 확률 p(w)를 식으로 쓰면 다음과 같다:

  <img src="https://latex.codecogs.com/gif.latex?%5Clog%20%7B%20%5Cleft%5C%7B%20p%28w%29%20%5Cright%5C%7D%20%7D%20%3D%5Csum%20_%7B%20d%3D1%20%7D%5E%7B%20D%20%7D%7B%20%5Csum%20_%7B%20j%3D1%20%7D%5E%7B%20V%20%7D%7B%20%7B%20n%20%7D%5E%7B%20jd%20%7D%5Clog%20%7B%20%5Cleft%5B%20%5Csum%20_%7B%20k%3D1%20%7D%5E%7B%20K%20%7D%7B%20%7B%20%5Ctheta%20%7D_%7B%20k%20%7D%5E%7B%20d%20%7D%7B%20%5Cphi%20%7D_%7B%20k%20%7D%5E%7B%20j%20%7D%20%7D%20%5Cright%5D%20%7D%20%7D%20%7D"/>

  * 모든 문서 내 등장하는 여러 토픽에 해당하는 단어의 분포를 계산하기 위해 합을 계산

* 이를 이용한 Perplexity 지표는 다음과 같이 구한다:

  <img src="https://latex.codecogs.com/gif.latex?Perplexity%28w%29%3Dexp%5Cleft%5B%20-%5Cfrac%20%7B%20log%5Cleft%5C%7B%20p%28w%29%20%5Cright%5C%7D%20%7D%7B%20%5Csum%20_%7B%20d%3D1%20%7D%5E%7B%20D%20%7D%7B%20%5Csum%20_%7B%20j%3D1%20%7D%5E%7B%20V%20%7D%7B%20%7B%20n%20%7D%5E%7B%20jd%20%7D%20%7D%20%7D%20%7D%20%5Cright%5D"/>



### 구현

