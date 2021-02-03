# 3. Bellman Equation

## 1) Bellman Expectation Equation

* Agent는 value function을 가지고 자신의 행동을 선택

* Value Function

  <img src="https://dnddnjs.gitbooks.io/rl/content/323da69f421b5fe15ce963cdfd804d40.png"/>

  * state-value function
  * action-value funtion

### Bellman equation for Value function

* 아래와 같이 다음 state와 현재 state의 value function 사이의 관계를 식으로 나타낸 것을 Bellam equation이라고 한다.

<img src="https://dnddnjs.gitbooks.io/rl/content/dfq.png"/>

* policy를 포함한 value function과 action value function도 Bellman equation의 형태로 표현할 수 있다.

<img src="https://dnddnjs.gitbooks.io/rl/content/18eba72dcfeafa6e6280055a95078ffa.png"/>

* 좀 더 직관적으로 살펴보기 위해 현재 state의 value function과 다음 state의 value function의 상관관계 식을 구해보자.

  * 이를 위해 state-action pair(어떤 state에서 어떤 행동을 한 상태를 하나의 state와 같이 생각하는 개념)을 고려해보자.

    1. state와 state-action 사이의 관계

       <img src="https://dnddnjs.gitbooks.io/rl/content/ddfdf.png"/>

       * 흰 점은 state를, 검은 점은 state에서 action을 선택한 상황을 의미
       * state에서 뻗어나가는 가지 => 가능한 action의 개수
       * 이때 V와 q의 관계는 policy로써 위 식처럼 표현이 가능하다.
         * 각 action을 취할 확률과 그 action을 해서 받는 expected return을 곱한 것의 합 => 현재 state의 value function!

    2. reward

       <img src="https://dnddnjs.gitbooks.io/rl/content/dfdfdfd.png"/>

       * r: reward
         * reward의 정의에 state와 action이 조건으로 들어가기 때문에 검은 점 밑에 표시된 것.
       * Deterministic한 환경이라면 action으로 하나의 가지만을 가지지만, 외부 요인에 따라 같은 state에서 같은 action을 취해도 다른 state로 갈 수도 있음.
         * 이를 state transition probability matrix라고 한다.
       * action-value function q_π(s, a)은 imediate reward + (action을 취했을 때 각 state로 갈 확률 * 그 위치에서의 value function) 으로 나타낼 수 있다.

  * 이 둘을 합치면 다음과 같이 표현할 수 있다.

    <img src="https://dnddnjs.gitbooks.io/rl/content/276f2082eb0ce52b5479f0678bdc24e0.png"/>

* 하지만 실제로 강화학습으로 무언가를 학습시킬 때, reward와 state transition probability를 미리 알 수는 없다.

  * 경험을 통해 알아가게 된다.
  * 만약 이러한 정보들을 다 안다면 MDP를 모두 안다고 표현하며, 이 정보들이 MDP의 모델이 된다.

* 강화학습의 가장 큰 특징은 MDP의 모델을 몰라도 학습이 가능하다는 점!

  * 따라서 reward function과 state transition probability를 모르고 학습하는 강화학습에서는 Bellman equation을 사용할 수 없다.



### Bellman equation for Q-function

* 같은 식을 action-value function에 대해 작성하고 그림을 보면 다음과 같다.

  <img src="https://dnddnjs.gitbooks.io/rl/content/acc6587c0d50511c8c21a32ce2d67d8a.png"/>



## 2) Bellman Optimality Equation

### Bellman expectation equation

* Backup: 미래의 값(next state-value function)으로 현재의 value function을 구함
  * One step / Multi step
  * Full-width(가능한 모든 다음 state의 value function을 사용해 백업)-Dynamic programming / sample(경험을 통한 백업)-reinforcement learning



### Optimal value function

* 강화학습의 목적: accumulative future reward를 최대로 하는 policy를 찾는 것

* optimal state-value function: 현재 state에서 policy에 따라 앞으로 받을 reward들이 달라지는데, 그 중 앞으로 가장 많은 reward를 받을 policy를 따랐을 때의 value function

  * optimal action-value function도 마찬가지로 현재 (s, a)에서 얻을 수 있는 최대의 value function

* 즉, 현재 environment에서 취할 수 있는 가장 높은 값의 reward 총합이 된다.

* 특히 두번째 optimal action-value function의 경우, 그 값을 안다면 단순히 높은 q값을 선택하면 되므로 최적화 문제를 해결했다, 라고 볼 수 있다.

  * Optimal policy는 (s, a)에서 action-value function이 가장 높은 action만을 고르기 때문에 deterministic 함.

    <img src="https://dnddnjs.gitbooks.io/rl/content/3334.png"/>

* 예시 그래프) 가운데 8이라고 써져 있는 state 입장에서, optimal action-value function의 값(q*) 중 가장 큰 8을 선택하면 그것이 optimal policy가 된다.

  <img src="https://dnddnjs.gitbooks.io/rl/content/4444.png"/>

### Bellman Optimality Equation

* Bellman Optimality equation: 위의 optimal value function 사이의 관계를 나타내주는 식

* 기존과 다른 점은 호 모양으로 표시된 "max"가 추가 된다는 점!

  <img src="https://dnddnjs.gitbooks.io/rl/content/555.png"/>

  <img src="https://dnddnjs.gitbooks.io/rl/content/1111.png"/>

* 두 diagram을 합치면 아래와 같이 표현할 수 있다.

  <img src="https://dnddnjs.gitbooks.io/rl/content/6565.png"/>