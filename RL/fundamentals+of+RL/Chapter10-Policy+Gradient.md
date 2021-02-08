# 10. Policy Gradient

## 1) Policy Gradient

### Value-based RL vs Policy-based RL

* 지금까지 다뤘던 방법들은 모두 "value-based" RL

  * Q라는 action-value function에 초점을 맞춰 Q-function을 구하고 그것을 토대로 policy를 구하는 방법

    <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/PG4.png"/></p>

  * 문제점:

    <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/PG7.png"/></p>

    * Unstable
      * Value-function을 바탕으로 policy를 계산하기 때문에 value-function이 약간만 달라져도 policy 자체가 크게 변화함
        * 전체적인 알고리즘이 수렴하는데 불안정성을 준다.
      * 이에 반해 policy-based RL을 사용해 policy 자체가 함수화가 되면 학습 하면서 조금씩 변화하는 value-function에 policy 또한 함께 조금씩 조정되며 안정적이고 부드럽게 수렴하게 됨.
    * Stochastic Policy
      * 때로는 Stochastic Policy가 Optimal Policy일 수도 있다
        * eg) 가위바위보: 가위, 바위, 보를 동등하게 1/3씩 내는 것이 optimal policy
          * value-based RL에서는 Q-function을 토대로 하나의 action만 선택하는 optimal policy를 학습하기 때문에 이러한 문제에는 적용시킬 수 없음.

* Policy-based RL: Policy 자체를 approximate하는 방법, 즉 policy 자체를 parameterize하는 것

  <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/PG5.png"/></p>

  

  <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/PG6.png"/></p>

  * 장점: 
    * 기존의 방법에 비해 수렴이 더 잘 됨
    * 가능한 action이 여러개(high-dimention)이거나 action 자체가 연속적인 경우에 효과적
  * 단점: 
    * 기존의 방법은 반드시 하나의 optimal한 action으로 수렴하는데 반해 policy gradient에서는 stochastic한 policy를 학습할 수 있음
    * local optimum에 빠질 수 있으며 policy가 evaluate하는 과정이 비효율적이고 variance가 높음

### Policy Objective Function

* eg) DQN에서는 prameter를 업데이트 하는데 TD error를 사용

  * Policy Gradient에서는 Objective Function이라는 것을 정의

* Objective Function를 정의하는 방법으로는 세가지가 있다:

  <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/PG9.png"/></p>

  1. start value: 처음 시작 state의 value function

  2. average value

  3. average reward per time-step: 각 time-step마다 받는 reward들의 expectation 값 사용

  * Stationary distribution: 각 state마다 정의되는 확률분포 값을 π라고 할 때, transition matrix T에 따라서 더이상 변화하지 않는 상태(πT = π)

* Policy Gradient에서 목표는 이 objective function을 최대화시키는 policy의 Parameter vector을 찾아내는 것

  * How? => Gradient Descent!

  <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/PG11.png"/></p>

  * Objective Function의 gradient를 구하는 방법으로는 다음 세가지 방법이 존재한다.

    * Finite Difference Policy Gradient

    * Monte-Carlo Policy Gradient

    * Actor-Critic Policy Gradient

      

## Finite Difference Policy Gradient

### Finite Difference Policy Gradient

<p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/PG12.png"/></p>

* 수치적으로 가장 간단하게 objective function의 gradient를 구할 수 있는 방법
* 만약 parameter vector가 5개의 dimension으로 이뤄져있다고 할 때, 각 parameter를 ε만큼 변화시켜보고 5개의 parameter에 대한 gradient를 각각 구하는 방법

* parameter space가 작을 때는 간단하지만 늘어날수록 비효율적이고 noisy한 방법
* policy가 미분 가능하지 않아도 작동한다는 장점이 있어 초기 policy gradient에서 사용되었다.



### Example: Training AIBO

* Finite Difference Policy Gradient를 이용해 Sony의 AIBO 학습

  * 로봇이 좀 더 빨리 걸을 수 있도록 튜닝하는데 강화학습을 사용

  <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/PG26.png"></p>

* 방식: 다리의 궤적 자체를 parameterize, 즉 궤적 자체를 policy로 보고 학습

  * objective function: 속도

  * gradient를 구하는 방법

    * parameter vector π (policy)

    * gradient를 estimate하기 위해 t개의 randomly generated policies를 만들어낸다.

      * 12개의 parameter에 대해 기존의 parameter보다 미세하게 랜덤으로 변화시킨 t개의 policy를 생성
      * 그 t개의 policy의 objective function(속도)를 측정
      * 12개 parameter 각각에 대해 average score을 계산해 update

      <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/PG28.png"/></p>

      

## 3) Monte-Carlo Policy Gradient: REINFORCE

* Finite Difference Policy gradient: numerical method
* Monte-Carlo Policy Gradient & Actor-Critic: analytical
  * analytical하게 gradient를 계산 = objective function에 직접 gradient를 취해준다는 것, 즉 policy가 미분 가능하다고 가정
  * episode마다 게산 => MC / time-step마다 계산 => Actor-critic

### Score Function

* analytical하게 gradient를 계산하기 위해 objective function에 gradient를 취하면 다음과 같다:

  <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/PG14.png"/></p>

  * objective function: average reward formulation

  * θ에 대해 gradient를 취하기 때문에 policy에만 gradient를 취하고 있는 것을 확인할 수 있다.

    * 이때 log가 생기는 이유는 다음과 같다.

      * <img src="https://latex.codecogs.com/gif.latex?%5Cnabla_%7B%5Ctheta%7D%20%5Cpi_%7B%5Ctheta%7D"/> 에 <img src="https://latex.codecogs.com/gif.latex?%5Cpi_%7B%5Ctheta%7D"/>를 곱하고 나누는 과정에서 아래와 같이 log의 미분 형태가 되기 때문에 <img src="https://latex.codecogs.com/gif.latex?%5Cpi_%7B%5Ctheta%7D"/> 의 gradient를 <img src="https://latex.codecogs.com/gif.latex?log%20%5Cpi_%7B%5Ctheta%7D"/> 의 gradient로 바꿀 수 있다.

        <img src="https://dnddnjs.gitbooks.io/rl/content/PG15.png"/> 

      * 이렇게 바꾸지 않으면 다음과 같은 식이 만들어진다.

        <p align="center"><img src="https://latex.codecogs.com/gif.latex?%5Csum_%7Bs%5Cin%20S%7D%5E%7B%7D%7B%20d%28s%29%20%7D%20%5Csum_%7B%20a%5Cin%20A%20%7D%5E%7B%20%7D%20%5Cnabla%20_%7B%20%5Ctheta%20%7D%5Cpi%20_%7B%20%5Ctheta%20%7D%28s%2Ca%29%7BR%7D_%7B%20s%2Ca%20%7D"/></p>

        * 이렇게 되면 <img src="https://latex.codecogs.com/gif.latex?%5Cpi_%7B%5Ctheta%7D"/> 가 없어졌기 때문에 expectation을 취해줄 수가 없다.

* 따라서 score function은 다음과 같이 정의한다:

  <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/PG13.png"/></p>

  

### Policy Gradient Theorem

* Gradient는 다음과 같다: <img src="https://dnddnjs.gitbooks.io/rl/content/PG17.png"/>

  <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/pg31.png"/></p>

  * p(x): policy

  * <img src="https://latex.codecogs.com/gif.latex?%5Cnabla_%7B%5Ctheta%7D%20log%5Cpi_%7B%5Ctheta%7D"/> 는 이 policy를 표현하는 parameter space에서의 gradient가 된다.

  * 여기에 reward r이라는 scalar값을 곱해줌으로써 어떤 방향으로 policy를 업데이트해줘야 하는지를 결정한다.

    * 이때 policy가 어디로 얼만큼 update 될 것인지 척도가 되는 scalar function으로 immediate reward만 사용하면 그 순간에 대한 정보밖에 모르기 때문에 제대로 학습이 되지 않을 가능성이 높다.

    * 따라서 immediate reward 대신에 자신이 한 행동에 대한 long-term reward인 action-value function을 사용하겠다는 것이 **Policy Gradient Theorem**

      <p align="center">
        <img src="https://dnddnjs.gitbooks.io/rl/content/PG18.png"/>
      </p>

      

      * r 대신 Q-function을 사용해 표현
      * [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)에 증명



### Stochastic Policy

* stochastic policy를 표현하기 위해 딥러닝에서 많이 사용되는 nonlinear 함수인 Sigmoid나 Softmax를 사용한다.

* Sigmoid: <img src="https://latex.codecogs.com/gif.latex?S%28t%29%3D%7B%5Cfrac%20%7B1%7D%7B1&plus;e%5E%7B-t%7D%7D%7D"/>

  <p align="center">
    <img src="https://dnddnjs.gitbooks.io/rl/content/PG32.png"/>
  </p>

  

  * Discrete action space의 경우: 만약 가능한 action = {rignt, left}일 때
    * 이 함수에서 나오는 값이 1에 가깝다면 left, 0에 가깝다면 right라는 식으로 설정해 stochastic policy를 표현할 수 있음
  * Continuous action space: 어떤 로봇의 controller에 0부터 100까지 control input을 줄 수 있을 때
    * sigmoid함수를 통해 0이 나오면 control input은 0, 1이 나오면 control input은 100을 주는 식으로 설정

* Softmax: <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/11b61d999176b3e8db6efe6632b7cc62fa4d4c53"/>

  * Discrete action space에서 action이 3개 이상이 되면 sigmoid 함수로 표현하기가 애매해짐
    * 이런 경우에는 Softmax 함수를 사용하는 것이 좋다.
  * Action이 i = 1 ~ n까지 있을 때, action probability를 위의 함수로 표현할 수 있다.



### Monte-Carlo Policy Gradient

* 기존에는 모든 state에 대해 action-value function을 알기 어렵기 때문에 approximation을 진행

  * 그렇다면 policy를 자체를 update하기 위해 사용할 action-value function은 어떻게 알아내야 할까? => **Monte-Carlo**

    * episode를 경험해보고 받았던 reward를 기억
    * episode가 끝난 다음에 각 state에 대한 return을 계산
      * 이 return 자체가 action-value function의 unbiased estimation!

    <p align="center"> 
      <img src="https://dnddnjs.gitbooks.io/rl/content/PG20.png"/>
    </p>

    * 각 eisode에 대해 loop를 돌며 parameter의 update가 일어난다.
      * parameter를 regression이 아니라 stochastic gradient descent를 사용해 한 스텝씩 update 해 나간다.

    

## 4) Actor-Critic Policy Gradient

* REINFOCEMENT 알고리즘:
  * Return을 사용하기 때문에 MC 고유의 문제인 high-variance가 발생한다.
  * episode 자체가 길 수도 있기 때문에 학습하는 시간이 오래 걸릴수도 있다.
  * 이를 해결하기 위해 parameter를 하나 더 사용해서 action-value function도 approximation!



### Actor & Critic

* Critic: action-value function을 approximate하는 **w**를 업데이트 / Actor: policy를 approximate하는 **θ**를 업데이트

<p align="center"> 
  <img src="https://dnddnjs.gitbooks.io/rl/content/PG21.png"/>
</p>



* Critic: action-value function을 통해 현재의 policy를 평가하는 역할을 수행
  * action을 해보고, 그 action의 action-value function이 높았으면 그 action을 할 확률을 높이도록 policy의 parameter를 업데이트
  * 판단 척도가 되는 action-value function 또한 처음에는 잘 모르기 때문에 학습을 해주기 위해 critic이 필요하다.
* Action-value function은 TD(0)을 사용해 업데이트

<p align="center"> 
  <img src="https://dnddnjs.gitbooks.io/rl/content/PG34.png"/>
</p>

* 위의 식은 action-value function을 linear하게 approximation했을 경우이다.
  * 매 time-step마다 업데이트를 진행하며, 업데이트 시 policy와 action-value function의 parameter를 동시에 업데이트



### Baseline

* Baseline: Actor-Ctiric말고 다르게 Variance문제를 해결하는 방법

  <p align="center"> 
    <img src="https://dnddnjs.gitbooks.io/rl/content/PG23.png"/>
  </p>

  

  * State-value fucntion을 일종의 평균으로 사용해 현재의 행동이 평균적으로 얻을 수 있는 value보다 얼마나 더 좋은지를 계산하도록 해 variance를 줄이는 것
  * 지금까지 해왔던 것 보다 좋으면 그 방향으로 업데이트, 아니면 반대방향으로 업데이트

* 이러한 advantage function의 사용은 variance를 상당히 개선시킬 수 있는 장점이 있지만 value function과 action-value function을 둘 다 approximation 해줘야 한다는 단점도 존재

  <p align="center"> 
    <img src="https://dnddnjs.gitbooks.io/rl/content/PG35.png"/>
  </p>

  
  * 하지면 action-value function이 immediate reward + value function이라는 것을 생각하면 결국 value function 하나만 approximate해도 되기 때문에 critic에 parameter를 두개 사용하는 비효율성을 개선할 수 있음

    <p align="center"> 
      <img src="https://dnddnjs.gitbooks.io/rl/content/PG36.png"/>
    </p>

    