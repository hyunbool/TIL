# 9. DQN(Deep Q-Networks)

## 1) Neural Network

### What is DQN

* RL의 작동 원리
  * Agent는 MDP을 통해 environment에 대한 이해를 하게 된다.
    * 이때 table 형태로 모든 state에 대한 action-value function 값을 저장하고 update하는 식으로 학습이 진행된다면 속도는 매우 느려질 것.
  * 이를 해결하기 위해 approximation을 하게 됨
    * nonlinear function approximator로 deep neural network가 있음.
* DeepRL: Action-value function(Q-value)나 policy를 approximate하는 방법으로 deep neural network를 택한 방식
* 특히 Action-value function을 approximate할 때 DNN을 사용하는 경우를 Deep Q-Networks(DQN)이라 한다.



# 2) Deep Q-Networks

* [Playing Atari with Deep Reinforcement Learning(2013)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) 에 소개된 방식


<p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/90-6.png"/></p>

* Contribution:
  * input data로 raw pixel을 받아온 점
  * 같은 agent로 여러 개의 게임에 적용되어 학습이 된다는 점
    * CNN을 사용해 게임 픽셀 데이터 자체를 학습 시킴으로써 게임마다 Agent를 설명해주지 않아도 여러 게임에 대해 한 agent로 학습이 가능해짐.
  * CNN을 function approximator로 사용
  * Experience Replay

* Deep Q-Network라는 개념이 처음 소개된 논문
  * DQN: action-value function approximator로 deep learning model 중 CNN을 사용한 기법

* 알고리즘:

  * experience replay를 사용

    * transition data들을 replay memory에 넣어놓고 매 time step마다 mini-batch를 랜덤으로 memory에서 꺼내서 update

      <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/dqn16.png"/></p>

  * psuedo code:

    <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/dqn17.png"/></p>

    * replay memory는 N개의 episide를 기억할 수 있음

      * N개가 넘어가면 오래된 episode부터 제외

    * episode마다 update 진행(1~M)

      * loss function 정의하고 그 gradient를 따라 업데이트

        * Q가 update 되어야 할 target을 <img src="https://latex.codecogs.com/gif.latex?r&plus;%5Cgamma%20maxQ%28s%27%2Ca%27%29"/> 로 두고 error를 계산해 GD 진행

          <p align="center"><img src="https://dnddnjs.gitbooks.io/rl/content/dqn18.png"/></p>

          





