# 8. Value Function Approximation

## 1) Value Function Approximation

### Tabular Methods

* 지금까지 살펴본 기법들은 action-value function을 Table로 만들어서 푸는 **Tabular Method**
  * We have so far assumed that our estimates of value functions are represented as a table with one entry for each state or for each state-action pair. This is a particularly clear and instructive case, but of course it is limited to tasks with small numbers of states and actions. The problem is not just the memory needed for large tables, but the time and data needed to fill them accurately. In other words, the key issue is that of **generalization**
  * 현재의 방법은 state나 action이 작을 때에만 적용이 가능
  * Table이 점점 더 커지면 이를 전부 기억할 메모리도 문제지만, 학습에 너무 많은 시간이 소요

<img src="https://dnddnjs.gitbooks.io/rl/content/apx1.png"/>

* 그동안 살펴본 방법으로는 위와 같은 문제를 해결할 수 없다.
  * Generalization을 하기 위해서는 새로운 기법이 필요하다.
    * 특히 실제 세상은 continuous state space이기 때문에 사실상 state가 무한대라고 할 수 있음



### Parametterizing value function

* Table로 작성하는 것이 아닌, w라는 새로운 변수를 사용해 value function을 함수화하자!

  <img src="https://dnddnjs.gitbooks.io/rl/content/apx2.png"/>

* state s가 함수의 input으로 들어가면 parameter w를 가진 함수가 action-value function을 output으로 내보낸다.

  <img src="https://dnddnjs.gitbooks.io/rl/content/apx3.png"/>

  * 지금부터는 학습을 통해 Q-function을 업데이트 시키는 것이 아닌, w라는 parameter를 업데이터하게 된다. => **function approximation**

    <img src="https://dnddnjs.gitbooks.io/rl/content/apx4.png"/>

    * 이 중 위의 두가지를 살펴보도록 한다.
      * 특히 최근의 강화학습은 대부분 딥러닝을 approximator로 사용하기 때문에 보통 Deep Reinforcement Learning이라고 부른다.



## 2) Stochastic Gradient Descent

### Gradient Descent

<img src="https://dnddnjs.gitbooks.io/rl/content/apx5.png"/>

* weight w를 가지는 함수 J(w)는 error를 최소화 하는 것을 목표로 한다.
  * update를 하기 위해서는 어느 방향으로 가야 이 error가 줄어드는지를 알아야 하는데, 이는 함수에 미분(gradient)을 취함으로써 알아낼 수 있다.

### Gradient Descent on RL

<img src="https://dnddnjs.gitbooks.io/rl/content/apx8.png"/>

* RL에서의 J(w): true value function <img src="https://latex.codecogs.com/gif.latex?v_%7B%5Cpi%7D%28S%29"/>과 approximate value function <img src="https://latex.codecogs.com/gif.latex?%5Chat%7Bv%7D%28s%2Cw%29"/> 의 Mean Squared Error

* Gradient Descent: (1) SGD (2) Batch 방법

  * Batch: 위와 같이 모든 state에서 true value function과의 error를 한번에 함수로 잡아 업데이트

    <img src="https://dnddnjs.gitbooks.io/rl/content/apx9.png"/>

  * SGD: expectation을 없애고 sampling으로 대체

    <img src="https://dnddnjs.gitbooks.io/rl/content/apx10.png"/>

* MC와 TD에서 처럼 True value function을 다른 것으로 대체할 수도 있다.

  * eg) Sample Return, TD target ...

  * Gt: return

    <img src="https://dnddnjs.gitbooks.io/rl/content/apx11.png"/>

## 3) Learning with Function Approximator

### Action-value function approximation

* Model-free가 되기 위해서는 action-value function을 사용해야 한다.

  <img src="https://dnddnjs.gitbooks.io/rl/content/apx12.png"/>

  * policy evaluation은 parameter의 update로 진행
  * policy improvement는 그렇게 update된 action-value function에 ε-greedy한 action을 취함으로써 진행

* Action-Value function Approximation

  <img src="https://dnddnjs.gitbooks.io/rl/content/apx13.png"/>

* True Value Function

  <img src="https://dnddnjs.gitbooks.io/rl/content/apx14.png"/>

### Example 

* eg) Mountain Car

  <img src="https://dnddnjs.gitbooks.io/rl/content/apx17.png"/>

  * 정상을 제외한 모든 곳은 time step마다 reward를 -1씩 받음
  * 따라서 Agent는 최대한 빠른 시간 안에 goal에 도달하는 것을 목표로 함
  * 바로 uphill을 할 추력이 차에게 없다고 가정
    * 차가 왔다 갔다 하면서 중력으로 가속시켜서 올라가야 한다.

* 많은 Control 문제에서와 마찬가지로 state는 위치와 속도로 정의된다.
  * 두개의 component를 가지기 때문에 state-space는 2차원으로 정의된다.
* optimal policy를 알아내기 위해서는 각 state의 value function을 알아야 하지만, state가 continuous하기 때문에 기존의 tablular methods로는 풀 수 없다.
  * 따라서 value function approximator를 통해 모든 state의 value function을 함수화해서 표현
  * 샘플링과 experience를 통해 value function을 학습해나간다.

<img src="https://dnddnjs.gitbooks.io/rl/content/apx15.png"/>

* 학습이 완료된 value function은 아래와 같이 표현할 수 있다.

  <img src="https://dnddnjs.gitbooks.io/rl/content/apx16.png"/>

* Agent가 하는 일은 각 state에서 가장 높은 value function을 가진 state로 이동하는 것!



### Batch Methods

* SGD를 통해 parameter를 업데이터할 때 생기는 문제점:

  <img src="https://dnddnjs.gitbooks.io/rl/content/apx21.png"/>

* Batch Methods: training data(Agent가 경험한 것들)를 모아서 한꺼번에 update

  * 한번에 업데이트하기 때문에 많은 데이터에 가장 잘 맞는 value function을 찾기가 어려움.
    * 이 때문에 SGD와 Batch의 중간 방법을 사용하는 경우도 존재
    * eg) mini-batch

* SGD의 문제점인 'experience data를 한번먼 사용하는 것이 비효율적임'은 한번만 사용하지 않고 여러번 사용하는 것으로 문제를 해결할 수 있다.

  * 그렇다면 어떻게 여러번 experience data를 활용할 것인가? => **experience replay**



### Experience Replay

<img src="https://dnddnjs.gitbooks.io/rl/content/apx20.png"/>

* replay memory를 만들어 agent가 경험했던 것들을 <img src="https://latex.codecogs.com/gif.latex?%28s_t%2C%20a_t%2C%20r_%7Bt&plus;1%7D%2C%20s_%7Bt&plus;1%7D%29"/> 로 각 time-step마다 저장

* action-value function의 parameter를 업데이트하는 것은 time-step마다 진행

  * 이때 하나의 transition에 대해서만 하는 것이 아니라 모아뒀던 transition을 replay memory에서 정해진 개수만큼 꺼내 mini-batch에 대해 update를 진행한다.

    