# 4. Dynamic Programming

* Dynaic Programming(DP): 완전한 모델이 environment로 주어졌을 때(MDP) 최적의 policy를 계산해낼 수 있는 알고리즘
  * *The term dynamic programming (DP) refers to a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as a Markov decision process (MDP)*

## 1) Policy Iteration

### Planning vs Learning

<img src="https://dnddnjs.gitbooks.io/rl/content/4225132.png"/>

* Planning vs Learning
  * Planning: environment의 모델을 알고서 문제를 푸는 것
  * Learing: environment의 모델은 모르지만 상호작용을 통해 문제를 푸는 것
* DP는 Planning으로서 environment의 모델(reward, state transition matrix)에 대해 안다는 전제로 문제를 해결하는 방법(Bellman)



### Prediction & Control

<img src="https://dnddnjs.gitbooks.io/rl/content/7788.png"/>

* DP는 두가지 step으로 나눌 수 있다: 1) Prediction 2) Control
  * 현재 optimal하지 않은 어떤 policy에 대해 value function을 구하고(prediction)
  * 현재의 value function을 토대로 더 나은 policy를 구해 이와 같은 과정은 반복해 optimal policy를 구하게 된다.



### Policy evaluation

<img src="https://dnddnjs.gitbooks.io/rl/content/111212.png"/>

* Policy evaluation: prediction 문제를 푸는 것, 현재 주어진 policy에 대한 true value function을 구하며 Bellman equation을 사용

  * 현재 policy를 가지고 one-step backup을 사용해 true value function을 구함

* 이전의 Bellman equation에 iteration 과정(k)이 추가된다.

  <img src="https://dnddnjs.gitbooks.io/rl/content/1601b1e72a52c39d2fc6447597f0ff3b.png"/>

  * 전체 MDP의 모든 state에 대해 동시에 한번씩 Bellman equation을 계산해 업데이트

* eg) 4*4 gridword policy evaluation 과정

  <img src="https://dnddnjs.gitbooks.io/rl/content/feewa.png"/>

  * state: 회색으로 표시된 terminal state / 14개의 non-terminal state

  * action: 상, 하, 좌, 우

  * time step이 지날 때 마다 -1의 reward를 받음

    * 따라서 agent는 reward를 최대로 해야하기 때문에 terminal state로 가능한 한 빨리 가려고 한다.
      * 이러한 policy를 계산해내는 것 ==> DP!

  * Evaluation: 현재 policy가 얼마나 좋은가를 판단하는 것

    * 판단 기준: 그 policy를 따라가게 될 경우 받게 될 value function

    * 처음 policy는 uniform random polic- 모든 state에서 똑같은 확률로 상하좌우로 움직이는 policy

      <img src="https://dnddnjs.gitbooks.io/rl/content/dp3.png"/> <img src="https://dnddnjs.gitbooks.io/rl/content/dp5.png"/>

      * k=1) V = 4 * 0.25(-1 + 0) = -1
      * k=2) (1,2)의 경우 V = 1 * 0.25(-1+0) + 3 * 0.25(-1 + -1) = -1.7
        * 위로 action을 취하면 벽에 부딪히기 때문에 자신의 state로 돌아옴
        * left, right, down의 action에 의해서 agent는 각각 -1의 value function을 가진 state에 도달



### Policy iteration

* improvement: 해당 policy에 대한 참 값을 얻었으면 이제 policy를 더 나은 policy로 update

  * 그래야 점점 optimal policy에 가까워질 수 있음

* improve하는 방법으로는 greedy improvement가 있음

  * 다음 state 중 가장 높은 value function을 가진 state로 가는 것(max를 취하는 것)

* Improvement를 반복하면서 optimal policy를 구하게 된다 => Policy Iteration!

  <img src="https://dnddnjs.gitbooks.io/rl/content/6d484ed095cba2cd7a8edf50b7e4e17e.png"/>

## 2) Value Iteration

### Value Iteration

* Policy Iteration과 다른 점은 Bellman Optimality Equation을 사용한다는 점
  * Bellman Optimality Equation: optimal value function들 사이의 관계식
  * Policy Iteration의 경우에는 evaluation 과정에서 수많은 계산을 해줘야 하는 단점이 있었지만, value iteration에서는 evaluation을 한번만 해줘도 됨.
    * 현재 value function을 계산 => update시 max를 취해 greedy하게 improve하는 효과

<img src="https://dnddnjs.gitbooks.io/rl/content/674af1b62041c9bfb73361222264c073.png"/>

### Sample Backup

<img src="https://dnddnjs.gitbooks.io/rl/content/vfgfda.png"/>

* DP => MDP에 대한 정보를 가지고 있어야 optimal policy를 구할 수 있기 때문에 MDP가 너무 크거나 전부 알지 못하는 경우 DP를 사용할 수 없음

  * 이 경우에 sample back-up 사용

* Sample Back-up

  * 샘플링을 통해 한 길만 가보고 그 정보를 토대로 value function를 업데이트

    * 계산이 효율적일 뿐만 아니라 model-free가 가능하다.

    * DP에서는 매 iteration마다 reward function과 state transition matrix를 알아야 함

    * Sample Backup의 경우 아래 그림과 같이 <S, A, R, S'>에서 실제 나온 reward와 sample transition으로 두개를 대체

      * 모델을 몰라도 optimal policy를 구할 수 있게 됨 => LEARNING!

      <img src="https://dnddnjs.gitbooks.io/rl/content/qqqq.png"/>