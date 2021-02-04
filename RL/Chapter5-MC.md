# 5. Monte-Carlo Methods

## 1) Monte-Carlo prediction

### Model-Free

* Dynamic Programming은 Model-based한 방법

  * 문제점
    * Full-width Backup => expensive computation
    * Full knowledge about Environment
  * 경우의 수가 많은 문제나 실제 task를 풀 수 없다.

* DP처럼 full-width backup을 하는 것이 아니라, 실제로 경험한 정보들을 가지고 update를 하는 sample backup 진행

  * environment에 대해 모든 것을 알 필요가 없다.

    * Environment의 모델을 모르고 학습을 하기 때문에 **Model-Free** 라고 할 수 있다!

      <img src="https://dnddnjs.gitbooks.io/rl/content/qqqq.png"/>

      * Model-free prediction: 현재의 policy를 바탕으로 움직여보며 sampling을 통해 value function을 update
      * Model-free control: 거기에 policy까지 update

* 샘플링을 통해 학습하는 model-free 방법에는 두가지가 있다.

  * Monte-Carlo: episode마다 update하는 방법
  * Temporal Difference: time step마다 update하는 방법



### Monte-Carlo

* Monte-Carlo

  * The term "Monte Carlo" is often used more broadly for **any estimation** method whose operation involves a significant **random component**. Here we use it specifically for methods based on **averaging complete returns**
  * 무엇인가를 랜덤하게 측정하는 것을 뜻하는 말이며, 강화학습에서는 최종 리턴에 대해 평균을 내는 방법을 의미한다.

* Monte-Carlo와 Temporal Difference로 갈리는 것은 value function을 추정하는 방법에 따라서이다.

* value function= expected accumulative future reward로서 현재 state에서 시작해 미래까지 받을 expected reward의 총합

  * episode를 끝까지 가본 후 받은 reward들로 각 state의 value function을 거꾸로 계산해보는 방법

    <img src="https://dnddnjs.gitbooks.io/rl/content/MC1.png"/>

    * initial state S1에서 시작해 terminal state St까지 현재 policy를 따라 움직임
      * 한 time step마다 reward를 받게됨
    * 이 reward들을 기억해두었다가 St가 되면 뒤돌아보며 각 state의 value function을 계산한다(Recall that the value function is the expected return)
      * 순간순간 받았던 reward들을 시간 순서대로 discount시켜 sample return을 구할 수 있다.



### First-Visit MC vs Every-Visit MC

* 위에서 설명한 것은 single episode의 경우

  * 하지만 multiple episode의 경우에는 한 episode마다 얻었던 return을 어떻게 계산해야할까?

    * MC -> 단순히 평균을 취해준다.
      * 한 episode에서 어떤 state에 대해 return을 계산
      * 다른 episode에서도 그 state를 지나가서 다시 새로운 return을 얻었을 경우 그 두개의 return을 평균
      * 또 해당 state을 지나가서 다시 새로운 return을 얻었을 경우 그 두개의 return에 대해 평균
      * 그 return들이 쌓일수록 true value function에 가까워지게 된다.

  * 그렇다면, 한 episode 내에서 어떤 state를 두 번 방문한다면?

    * First-visit Monte-Carlo Policy evaluation

      <img src="https://dnddnjs.gitbooks.io/rl/content/MC2.png"/>

      * 처음 방문한 state만 인정
      * 두번째부터의 그 state 방문에 대해서는 return 계산하지 않음

    * Every-visit Monte-Carlo Policy evaluation 

      * 방문할 때 마다 따로따로 return을 계산

    * 두 방법 모두 true value function으로 수렴한다.

      * 해당 강의에서는 First-visit MC에 대해서만 다룸



### Incremental Mean

* 하나 하나 더해가며 return에 대한 평균을 계산 => Incremental Mean의 식으로 표현할 수 있다.

  <img src="https://dnddnjs.gitbooks.io/rl/content/MC3.png"/>

* Incremental Mean을 위의 First-visit MC에 적용

  <img src="https://dnddnjs.gitbooks.io/rl/content/MC4.png"/>

  * 분모의 N(S_t)가 점점 무한대로 발산하게 되므로 이를 α로 고정시켜 계산한다.
    * 맨 처음 정보들에 대해 가중치를 덜 주는 형태가 된다.
    * 매 episode마다 새로운 policy를 사용하는 non-stationary problem이므로 update하는 상수를 일정하게 고정시킨다.



### Backup Diagram

<img src="https://dnddnjs.gitbooks.io/rl/content/MC5.png"/>

| DP                                                           | MC                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| one step만 표시                                              | terminal state까지 쭉 이어짐                                 |
| one-step backup에서 그 다음으로 가능한 모든 state들로 가지가 뻗음 | 샘플링을 하기 때문에 하나의 가지로 terminal state까지 이어짐 |

* Monte-Carlo => random process를 포함한 방법
  * episode 마다 update를 하기 때문에 처음 시작이 어디였냐에 따라 같은 state에서 어디로 가는지에 따라 전혀 다른 experiencerk ehlsek.
  * 따라서 variance가 높지만 어딘가에 치우치는 경향은 적어 bias는 낮다.





## 2) Monte-Carlo Control

### Monte-Carlo Policy Iteration

* Monte-Carlo Policy Iteration = Monte-Carlo Policy Evaluation + Policy Improvement

* DP의 Policy Iteration

  <img src="https://dnddnjs.gitbooks.io/rl/content/MC8.png"/>

  * 현재 policy를 토대로 value function을 iterative하게 계산해 true value function에 수렴할 때 까지 policy를 evaluation(prediction)
  * 그 value function을 토대로 greedy하게 policy를 improve
  * optimal policy를 얻을 때 까지 반복

* MC도 마찬가지로 진행한다.

  <img src="https://dnddnjs.gitbooks.io/rl/content/MC9.png"/>

### Monte-Carlo Control

* 하지만 Monte-Carlo Policy Iteration의 경우 세가지 문제점이 존재한다.

  1. Value function 

     * MC에서 policy를 evaluation하는데 value function을 사용한다.

       * 하지만 이렇게 value function을 사용하면 policy를 greedy하게 improve시킬 때 문제가 발생한다.

     * MC => Model-free를 위해서 사용

       <img src="https://dnddnjs.gitbooks.io/rl/content/MC10.png"/>

       * 하지만 value function으로 policy를 improve하려면 MDP의 모델을 알아야 한다.
         * reward와 transition을 알아야 policy를 계산할 수 있다.
         * 따라서 value function 대신에 action value function을 사용한다.
           * model-free로 계산 가능!

  2. Exploration

     * 현재 policy improve를 위해 greedy policy improvement를 사용

       * 하지만 이 경우 local optimum에 빠져버릴 수 있음
         * 충분히 exploration을 하지 못했기 때문에 현재의 action a가 높은 value function을 가진다고 측정이 되어 더 높은 value function을 가질 수도 있는 action b를 배제해버리게 된다.

     * 이에 대한 대안으로 Epsilon greedy policy improvement를 한다.

       <img src="https://dnddnjs.gitbooks.io/rl/content/MC11.png"/>

       * 일정 확률(epsilon)로 현재 상태에서 가장 높은 가치를 가지지 않는 다른 action을 하도록 한다.
       * 선택할 수 있는 action이 m개 있을 경우, greedy action과 다른 action들을 위와 같은 확률로 나눠서 선택한다.

       

  3. Policy Iteration

     * Evaluation 과정이 true value function으로 수렴할 때 까지 Policy Iteration 진행

       * 이렇게 하지 않고 한번 evaluation한 다음 policy improve를 해도 optimal로 갈 수 있다. => Value Iteration

     * MC에서도 evaluation 과정을 줄여 Monte-Carlo policy iteration => Monte-Carlo Control이 된다.

       <img src="https://dnddnjs.gitbooks.io/rl/content/MC12.png"/>



### GLIE(Greedy in the Limit with Infinite Exploration)

<img src="https://dnddnjs.gitbooks.io/rl/content/MC13.png"/>

* 학습을 해나가면서 충분한 exploration을 진행했다면 greedy policy에 수렴하는 것을 말한다.
* eg) epsilon greedy policy
  * greedy하게 하나의 action만 선택하는 것이 아니기 때문에 GLIE하지 않음.
  * 하지만 epsilon이 시간에 따라 0으로 수렴한다면, epsilon greedy 또한 GLIE가 될 수 있다.