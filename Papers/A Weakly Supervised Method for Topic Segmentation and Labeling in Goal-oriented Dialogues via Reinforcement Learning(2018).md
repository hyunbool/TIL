# A Weakly Supervised Method for Topic Segmentation and Labeling in Goal-oriented Dialogues via Reinforcement Learning(2018)

## Abstract

* 대화 발화문에서 topic boundary를 찾고, 해당 발화문에 topic label을 부여하기 위해 강화학습을 사용한 topic segmentation과 goal-oriented dialogues를 라벨링하는 기법을 제안한다.
* goal-oriented customer service dialogue에는 세가지의 issue가 존재한다.
  * informality
  * local topic continuity
  * global topic structure
* Weakly supervised setting에서 해당 task를 살펴보고, sequential decision problem으로 공식화 시켰다.
* 제안된 방법은 informality issue를 위한 state representation network와 local topic continuiuty와 global topic structure 모델에 대한 reward를 부여하는 policy network로 구성된다.
  * 두가지 네트워크를 학습하고, policy에 대해 warm-start를 할 수 있도록 
    * 자동으로 데이터의 keyword를 annotate
  * 그런 다음 noisy data로 네트워크를 pre-train
  * 더 좋은 policy를 얻기 위한 state representation을 찾아내기 위해 current policy로 데이터를 refine하는 작업을 계속 진행한다.
* Sota baseline보다 본 논문의 weakly supervised 방법이 더 좋음을 실험으로 밝혔다.



## 1. Introduction

* 발화의 토픽 boundaty를 찾고, 각 발화마다 토픽 라벨을 할당하기 위해 Goal-oriented dialogue의 토픽 구조를 분석하는 것을 목표로 한다.
* 다른 일반 텍스트들에 비해 goal-oriented dialogue는 다음과 같은 세가지 특징을 가진다:
  * informality: 단편화되거나 불완전하거나 오타나 구어체가 포함된 문장
  * local topic continuity:  하나의 문제에 대해 얘기를 진행할 때는 대부분 하나의 토픽에 대해서만 얘기한다.
  * global topic structure: 각 dialogue 세션은 딱 떨어지는 boundary를 가지고 있으며, 토픽 간 cross-transition은 적고 cohesion은 높은 특징을 지닌다.
* 하지만 기존의 방법들은 위의 특징들을 제대로 반영하지 못하고 있다.
  * local topic continuiuty를 해결하기 위해 기존에는 어휘적 cohesion이나 phrase similarity를 이용해 왔다.
    * 문장 레벨의 dependency를 고려하지 못하고, 문맥을 제대로 요약하지 못한다.
    * 응집된 local topic assignment를 만들 수 없으며 fragmented segment를 만들어낸다.
  * 인접한 발언들 간 discourse dependency를 알아내기 위한 연구도 있었지만, 다이얼로그 시스템에서 global topic structure를 모델링 하는 것은 아직까지 연구된 적이 없음.
  * Fully supervised method들은 annotation을 진행하는 비용이 너무 크기 때문에 큰 데이터셋에는 적합하지 않음.
* 따라서 이 논문에서는 policy gradient reinforcement learning을 이용해 위의 세가지 특징을 고려하는 방법을 제안함.
  * topic segmentation과 labeling을 sequential decision problem으로 봄
    * 각 발언에 대해 sequential하게 토픽을 부여
    * 이전의 decision이 현재와 미래의 decision에 영향을 주기 때문
  * local topic continuity를 encourage하기 위해 intermediate reward를 정의
    * labeling 관점에서 local topic 간의 응집력을 높이는 역할을 한다.
  * 모든 sequential decision이 만들어 지고 나면, delayed reward를 이용해 한 세션에 대한 global topic structure를 측정한다.
    * delayed reward는 높은 segment 내 similarity와 낮은 segment 사이의 similarity를 선호한다.
  * informality 문제를 해결하기 위해서는 hierarchical LSTM을 사용한다. HLSTM을 이용해 단어 레벨과 문장 레벨 dependency를 포착하기 위한 state representation을 만들어 낸다.
    * HLSTM => 단순히 단어나 구절에 대한 유사성을 사용하는 것 보다 hitorical information을 더 잘 요약한다.
      * 따라서 HLSTM는 informality를 해결할 수 있을 뿐만 아니라 content의 관점에서 local topic continuity를 설명할 수 있다. 

* State representation network와 policy network로 구성된다.
  * 본 논문의 방법에서는 state representation을 만들어내는 것이 매우 중요하지만 라벨링 데이터 없이 좋은 representation을 학습해내기란 매우 어렵다.
  * policy network에서도 reward만 가지고 토픽 라벨링을 진행하는 것은 매우 어렵다.
  * 따라서 state representation을 학습하고, policy들이 topic을 detect하는데 warm-start를 제공해주기 위해서는 라벨링 된 데이터가 꼭 필요하다.
    * 하지만 큰 사이즈의 데이터에서 이런 라벨링을 진행하는 것은 매우 비용이 높다.
    * unsupervised method를 사용해 latent topic을 할당하는 방법도 존재하지만, 그렇게 만들어진 topic들은 간접적이고 task에 대한 직접적인 해석이라 보기 어렵다.
    * 따라서 본 논문에서는 noisy labeling을 이용한다.