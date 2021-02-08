# 7. Off-Policy Control

## 1) Importance Sampling

* 그동안 살펴보았던 Monte-Carlo Control과 Temporal를 Difference Control은 on-policy reinforcement learning



### On-Policy vs Off-Policy

* eg) Sarsa

  <img src="https://dnddnjs.gitbooks.io/rl/content/TD25.png"/>

  * '*Choose A from S using Policy derived from Q*', '*Choose A' from S' using Policy derived from Q*'와 같이 action을 선택할 때 **Policy** 를 이용함
    * 즉, 현재 policy대로 움직이면서 그 policy를 평가한다.
    * 그 평가를 토대로 각 step마다 Q function을 업데이트
  * 따라서, 현재 policy 위에서 control(prediction + policy + improvement)을 하기 때문에 **on-policy**

* On-Policy의 단점: exploration

  * 현재 알고있는 정보에 대해 greedy로 policy를 정해버리게 되면 optimal에 가지 못할 확률이 커짐
    * 따라서 Agent에게는 항상 exploration이 필요하다.

* Off-Policy: action을 수행하는 policy와 학습하는 policy를 분리시킨 것!

  <img src="https://dnddnjs.gitbooks.io/rl/content/Off1.png"/>

  * 장점:
    * 다른 agent나 사람을 관찰하고 그로부터 학습 가능
    * 이전의 policy를 재활용해 학습할 수 있음
    * exploration을 계속 하면서도 optimal한 policy를 학습할 수 있다(Q-learning)
    * 하나의 policy를 따르면서 여러개의 policy를 학습할 수 있다.

### Importance Sampling

* 다른 policy로부터 현재의 policy를 학습할 수 있다는 근거는 무엇일까? => **Importance Sampling**
* Importance Sampling: 다른 분포에서 샘플링 된 값을 가지고 구하고자 하는 분포에서 기댓값을 추정하는 방법
  * p와 q라는 서로 다른 distribution이 있을 때, q라는 distribution에서 진행함에도 불구하고 p로 추정하는 것 처럼 할 수 있다.
  * RL에서 policy가 다르면 state의 distribution이 달라지는데, 다른 policy를 통해 얻어진 sample을 이용해 Q 값을 추정하게 된다.

<img src="https://dnddnjs.gitbooks.io/rl/content/off3.png"/>

* f(X)를 value function이라 생각하면, RL에서는 이 value function을 expected future reward로 보고 계속 추정해 나간다.
  * P(X)라는, current policy로부터 만들어진 distribution으로부터 f(X)를 학습
* 이때 다른 Q라는 distribution을 따르면서도 똑같이 학습이 가능하다.
  * <img src="https://latex.codecogs.com/gif.latex?%5Cfrac%7BQ%28X%29%7D%7BQ%28X%29%7D"/> 를 취해주면 된다.

* Off-Policy 또한 MC와 TD로 나눌 수 있다.

  * Off-Policy MC:

    <img src="https://dnddnjs.gitbooks.io/rl/content/off4.png"/>

    * 에피스드가 끝나고 return을 계산할 때 다음과 같이 식을 변형해준다.
      * 각 step에 reward를 받게 된건 μ라는 policy를 따라 얻었던 것이기 때문에 매 step마다 <img src="https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpi%20%7D%7B%5Cmu%20%7D"/> 를 해주어야 한다.
    * 따라서 MC에 Off-Policy를 적용시키는 것은 그렇게 좋은 아이디어는 아님.

  * Off-Policy TD:

    <img src="https://dnddnjs.gitbooks.io/rl/content/off5.png"/>

    * Importance Sampling을 한번만 수행하면 된다.
    * MC에 비해 Variance가 낮아지긴 했지만, 여전히 높은 Variance를 가진다.

## 2) Q-Learning

### Q-Learning

<img src="https://dnddnjs.gitbooks.io/rl/content/off6.png"/>

* 현재 state S는 behavior policy를 따라 action을 선택하고, 다음 state의 action을 선택하는데는 alternative policy를 사용한다.
  * 더이상 Importance Sampling이 필요하지 않게 된다.
* 이전의 Off-Policy들에서는 value function을 사용했지만, 여기서는 action-value function을 사용한다.



### Off-Policy Control with Q-Learning

<img src="https://dnddnjs.gitbooks.io/rl/content/off7.png"/>

* Q-Learning 알고리즘 중에는 아래의 알고리즘이 가장 유명하다:
  * Behavior policy로는 **ε-greedy w.r.t. Q(s, a)**, Target policy로는 **greedy w.r.t. Q(s, a)**
  * exploratory policy를 따르면서도 optimal policy를 학습할 수 있게 된다.
    * greedy한 policy로 학습을 진행하면 수렴은 빠름 but 충분한 탐험은 하지 않았기 때문에 local에 빠지기 쉬움
    * 탐험을 위해 ε-greedy policy를 사용하면 탐험은 계속 할 수 있지만 수렴 속도가 느려져서 학습 속도가 느려지게 됨
    * 이를 해결하기 위해 다음과 같은 Q-Learning이나 ε을 시간에 따라 delay 시키는 방법을 사용한다.

* Q-function은 다음과 같이 update 된다:

  <img src="https://dnddnjs.gitbooks.io/rl/content/off8.png"/>

* 알고리즘

  <img src="https://dnddnjs.gitbooks.io/rl/content/off9.png"/>



### Sarsa vs Q-Learning

<img src="https://dnddnjs.gitbooks.io/rl/content/off10.png"/>

* 목표: S라는 start state에서 시작해 Goal까지 가는 optimal path를 찾자!
  * Cliff에 빠져버리면 -100의 reward를 받으며, time-step마다 -1씩 reward를 받는다
  * 절벽에 빠지지 않고 목표까지 가능한 한 빠르게 도착하는 것이 목표
* Sarsa와 Q-Learning 모두 다 ε-greedy한 policy로 움직이며, 그러다가 절벽에 빠져버리는 경우도 발생한다.
  * Sarsa: on-policy이기 때문에 절벽에 빠져버리게 되면 그 주변 상태들의 value가 낮다고 판단한다.
    * 이로 인해 정작 optimal로 수렴하지 못하는 현상이 발생한다.
  * Q-Learning: 그에 반해 절벽에 빠져버릴지라도, 자신이 직접 체험한 결과가 아니라 greedy한 policy를 사용한 Q-function을 이용해 update
    * 따라서 절벽 근처의 길도 Q-Learning은 optimal path라고 판단할 수 있다.
* 이렇게 많은 문제에서 Q-Learning이 더 효율적으로 문제를 풀수 있었기 때문에 강화학습에서는 Q-Learning이 기본적인 알고리즘으로 자리 잡게 된다.