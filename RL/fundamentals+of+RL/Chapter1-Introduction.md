# 1. Introduction

## 1.1 What is Reinforcement Learning

* [강화 학습]([https://ko.wikipedia.org/wiki/%EA%B0%95%ED%99%94_%ED%95%99%EC%8A%B5](https://ko.wikipedia.org/wiki/강화_학습)): 기계학습이 다루는 문제 중에서 다음과 같이 기술 되는 것을 다룬다. 어떤 환경을 탐색하는 에이전트가 현재의 상태를 인식하여 **어떤 행동을 취한다**. 그러면 그 에이전트는 **환경으로부터 포상을 얻게 된다**. 포상은 양수와 음수 둘 다 가능하다. 강화 학습의 알고리즘은 그 에이전트가 앞으로 **누적될 포상을 최대화하는 일련의 행동으로 정의되는 정책**을 찾는 방법

  * 에이전트(Agent)가 환경 속으로 들어가서 경험을 통해 학습하는 것!

* Sutton 교수님의 *Introduction to Reinforcement Learning* 중

  > Reinforcement learning is defined not by characterizing learning methods, but by characterizing a learning problem.

  * 강화학습은 학습하는 '방식'이 아니라 무엇을 학습해야 하는지에 대한 '문제'로 정의할 수 있다.

* Machine Learning의 세가지 범주

  * Supervised Learning
  * Unsupersived Learning
  * Reinforcement Learning
    * Trial and Error
      * 환경과 상호작용하며 직접 해보면서 자신을 조정해 나가게 된다!
    * Delayed Reward
      * 강화 학습이 다루는 문제에는 '시간'이라는 개념이 포함됨
      * 강화 학습은 시간의 순서가 있는 문제를 풀기 때문에 현재 행동으로 인한 환경의 반응이 늦어질 수 있음
        * 혹은 다른 행동과 합해져 더 좋은 반을을 받아낼 수도!
      * 따라서 환경이 반응할 때 까지 일련에 취했던 행동 중 어떤 행동이 좋은 행동이었는지 판단하기 어려운 점이 존재

## 1.2 History

* 강화 학습의 시작: Trial and error & Optimal control
  * Trial and Error
    * 스키너 상자 실험
      * 자신이 한 행동에 따른 보상으로 인해 더 좋은 보상을 받는 행동을 하도록 학습
  * Optimal control
    * 어떤 비용함수의 비용을 최소화하도록 컨트롤러를 디자인하는 것
    * Bellman equation: optimal control 문제 해결 => Dynamic Programming
  * 이러한 두가지가 합쳐지면서 강화학습 탄생
* Temporal difference Learning, Q Learning으로 발전해 오다가 최근 딥러닝과의 조합으로 엄청난 성과를 내고 있음
  * 알파고: Policy gradient with Monte-carlo Tree Search



## 1.3 Example

* 논문 *\<Playing atari with deep reinforcement learning\>*
  * 강화학습 + 딥러닝으로 atari 학습

<figure>
  <img src="https://dnddnjs.gitbooks.io/rl/content/90-6.png"/>
  <figcaption>출처: https://dnddnjs.gitbooks.io/rl/content/example.html</figcaption>
</figure>

* 강화 학습의 학습 대상: Agent
  * 처음에는 랜덤하게 움직임
  * 그러다가 게임의 점수가 올라가게 되면 Agent는 이 행동을 하면 점수가 올라간다고 판단해 해당 행동을 하도록 학습
  * 이때 단순히 현재의 점수만을 높이는 것이 아니라 시간이 지날수록 한 에피소드 동안 받는 점수를 최대화 시키려고 함
    * 이때 에이전트의 일련의 연속된 행동을 정책(policy)라고 함
  * 높은 점수 => 어떤 한 행동의 결과라기 보다는 각 상황에 맞는 행동들의 조합
    * 따라서 에이전트는 이런 정책을 더 높은 점수를 받는 쪽으로 변화

* Deep Reinforcement Learning의 경우 데이터의 숫자를 일일이 저장해 그것을 수행하는 것이 아니라 함수의 형태로 만들어 정확하지 않더라도 효율적으로 학습할 수 있는 방법 사용 ==> Approximation

  