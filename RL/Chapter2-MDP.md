# Chapter 2. Markov Decision Process

## 1. Markov Decision Process

<img src="https://dnddnjs.gitbooks.io/rl/content/216050d8baf8170c242d70f2e19803fa.png">

* 처음 상태에서 시작해 현재 상태까지 올 확률(3.4)이 직전 상태에서 현재 상태까지 올 확률(3.5)과 같을 때, state는 Markov하다고 말할 수 있다.

  * 강화학습은 기본적으로 MDP로 정의되는 문제를 풀기 때문에 state가 Markov하다고 가정하고 접근
    * 이는 절대적인 것은 아님(eg. Non-Markovian MDP ...)
  * 따라서 강화 학습에서는 value라는 어떤 가치가 현재의 state 함수로 표현되며 이 state가 Markov하다고 가정됨

* MDP는 state, action, state transition, probability matrix, reward, discount factor로 이뤄진 문제

  

  <figure>
    <img src="https://dnddnjs.gitbooks.io/rl/content/9864ef6a012bcbff9249a3805b06035d.png">
    <figcaption>출처: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html</figcaption>
  </figure>

  * **State: 에이전트가 인식하는 자신의 상태**

    * eg) "나는 방에 있어"라고 인식하는 과정에서 "방"이 state

  * **Action: 에어전트가 environment에서 특정 state에 갔을 때 지시하는 행동**

    * 에이전트는 action을 취함으로써 자신의 state를 변화
    * 로봇에서 Controller라고 불리는 부분

  * **State transition probability matrix: state s에서 action a를 취할 때 s'에 도착할 확률**

    <img src="https://dnddnjs.gitbooks.io/rl/content/f834cffade7ab13dcd32530fb3576db2.png"/>

    * 에이전트가 어떤 action을 취했을 경우, state가 deterministic하게 정해지는 것이 아니라 확률적으로 정해지는 것

  * **[Markov Chain](https://ko.wikipedia.org/wiki/%EB%A7%88%EB%A5%B4%EC%BD%94%ED%94%84_%EC%97%B0%EC%87%84): 과거와 현재 상태가 주어졌을 때의 미래 상태의 조건부 확률분포가 과거 상태와는 독립적으로 현재 상태에 의해서만 결정**

  * **Reward: state s에서 action a를 취할 때 얻을 수 있는 reward**

    <img src="https://dnddnjs.gitbooks.io/rl/content/af927db4928fa1c9c68c133ea73e0737.png"/>

    * 강화학습에서는 정답 혹은 환경에 대한 사전 지식이 없기 때문에 이 reward를 통해 에이전트가 학습
    * 에이전트는 현재의 reward 뿐만 아니라 이후 얻는 reward까지 고려

  * **Discount Factor**

    * 각 state마다 받았던 reward을 단순히 더해나가게 되면 문제가 발생

      * 점수를 계속 더해나가다 보면 그 값은 무한대로 발산함. 수학에서 무한대는 크기 비교를 할 수 없음
      * 서로 다른 policy에 대해 점수가 같은 경우 어떤 경우가 더 나은 것인지 판단할 수 없음

    * 시간에 따라 reward의 가치를 다르게 해 계산하는 방법

      <img src="https://dnddnjs.gitbooks.io/rl/content/7983adbb6486e6d5c6972fbba09e86c1.png"/>

* 이렇게 에이전트는 action을 취하고 state를 옮기고 reward를 받으면서 환경과 상호작용을 함

  <img src="https://dnddnjs.gitbooks.io/rl/content/da301af067262a7d688e281d4bade22f.png"/>

* **Policy: 어떤 state에서 어떤 action을 할 지에 대한 확률분포**

  * 강화학습의 목적은 optimal policy를 찾는 것

    <img src="https://dnddnjs.gitbooks.io/rl/content/b256481449d77879cff9109fbecb08d1.png"/>

    

* MDP Graph: state 사이의 transition 대신 action을 통한 state의 transition과 reward로 표현

  <img src="https://dnddnjs.gitbooks.io/rl/content/1.png"/>



## 1.2 Value Function

* value function: 기대값에 의해 reward를 예측하는 방법

### State-value function

* Return

  <img src="https://dnddnjs.gitbooks.io/rl/content/2f32323a0ff14183c045cfb04744ab73.png">

  * Agent가 state1에 있고, 끝이 있는 episode라고 가정했을 때, action을 취해가면서 이동했을 때 받은 reward의 총합을 **return**이라고 할 수 있다.

* 이 return의 기대값이 state-value function!

  <img src="https://dnddnjs.gitbooks.io/rl/content/4885d4877f3115bb054016dbd00e14ea.png">

  * 즉 어떤 state s의 가치라고 말할 수 있다.

  * Agent -> 다음에 이동할 수 있는 state의 가치를 보고 높은 가치를 가지는 state로 이동

    * 따라서 정확하고 효율적인 value function를 구하는 것이 매우 중요하다.

  * Value function을 계산하는데 있어 agent의 일련의 action(policy)을 반드시 고려해주어야 한다. 

    * 위의 식처럼 한가지 state만 고려할 수 없음

  * policy에 대한 value function

    <img src="https://dnddnjs.gitbooks.io/rl/content/232.png">

    * 대부분의 강화학습 알고리즘은 이 value function을 얼마나 잘 계산하는지가 중요한 역할을 한다.
      * bias되지 않고, 분산이 낮으며, true에 가까우며, 효율적으로 빠른 시간 안에 수렴해야 한다.

### Action-value Function

* Action: 어떤 state에서 할 수 있는 행동

  * state의 가치는 그 state에서 어떤 action을 했는지에 따라 달라지는 reward에 대한 정보를 포함한다.
  * Agent의 입장에서 다음에 취할 행동은 그 다음에 이동할 수 있는 state들의 value function으로 판단
    * 그러려면 다음 state에 대한 정보를 다 알아야 함
    * 그 state로 가려면 어떻게 해야하는지도 알아야 함
  * w그렇기 때문에 action에 대한 value function도 알아야 한다.

* Action-value function: 어떤 state s에서 action a를 취할 경우 받을 return에 대한 기대값

  <img src="https://dnddnjs.gitbooks.io/rl/content/e7b067d294a64c295cd120d1cdf33e20.png"/>

  * 어떤 행동을 할 지 이 함수의 값을 보고 판단할 수 있기 때문에 다음 state의 value function을 알고, 어떤 행동을 했을 때 해당 state에 갈 수 있는지에 대한 확률을 알지 않아도 된다.
  * Q-value라고도 한다. q-learning이나 deep q-network에서 사용되는 q가 이를 의미한다.



