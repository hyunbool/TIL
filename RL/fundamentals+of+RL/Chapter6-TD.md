# 6. Temporal Difference Learning

## 1) TD Prediction

### Temporal Difference

* MC= Model-Free Control
  * Model-Free라는 점에서 강화학습이지만, 단점도 존재.
    * 온라인으로 바로바로 학습할 수 없음
    * 꼭 terminal state가 존재하는 episode여야 함
    * episode가 길 경우에는 학습하기 어려움
  * episode가 끝나지 않더라도 DP처럼 time step마다 학습할 수 있는 방법? **Temporal Difference**
* Temporal Difference
  * If one had to identify one idea as central and novel to reinforcement learning, it would undoubtedly be temporal-difference (TD) learning. TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment's dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap)
  * TD는 MC와 DP를 섞은 것으로, MC처럼 raw experience로부터 학습할 수 있지만, DP처럼 time step마다 학습할 수 있는 방법
    * TD: 현재의 value function을 계산하는데 앞선 주변 state들의 value function을 사용 => Bellman Equation!



### TD

* MC의 incremental mean

  * return을 사용해 update

* TD(0): return Gt 대신 <img src="https://latex.codecogs.com/gif.latex?%7B%20R%20%7D_%7B%20t&plus;1%20%7D%20&plus;%20%5Cgamma%20V%28%7B%20S%20%7D_%7B%20t&plus;1%20%7D%29"/> 를 사용한다.

  * Temporal Difference Learning 방법에는 여러가지가 있는데 그 중 가장 간단한 방법

  * <img src="https://latex.codecogs.com/gif.latex?%7B%20R%20%7D_%7B%20t&plus;1%20%7D%20&plus;%20%5Cgamma%20V%28%7B%20S%20%7D_%7B%20t&plus;1%20%7D%29"/> 를 TD target이라고 부르며, 이 타겟과 현재의 value function과의 차이를 TD error라고 부른다.

    <img src="https://dnddnjs.gitbooks.io/rl/content/TD1.png"/>

  * TD(0)의 backup diagram은 다음과 같다:

    <img src="https://dnddnjs.gitbooks.io/rl/content/TD2.png"/>



### MC vs TD

* eg) 직장에서 집까지 가는 episode에 대한 prediction

  * state: 차가 막힘, 비가 옴, 고속도로를 나감, 집에 도착 ...

  * 각 state에 있을때 앞으로 얼마나 더 걸릴지에 대해 agent가 predict한 것에 대한 비교

    <img src="https://dnddnjs.gitbooks.io/rl/content/TD3.png"/>

  * value function: 어떤 state에서 앞으로 집으로 가기 까지 얼마나 걸릴지

    * agent는 value function을 predict

  * 시간이 지날 때 마다 실재로 흐른 시간이 reward

<img src="https://dnddnjs.gitbooks.io/rl/content/TD4.png"/>

* MC: 일단 다 도착한 다음 각각의 state에서 예측했던 value function과 실제로 받은 return을 비교해 update
* TD: 아직 도착하지 않아 얼마나 걸릴지는 정확히 모르지만, 한 스텝동안 지났던 시간을 토대로 value function을 update
  * 실제로 도착하지 않아도, final outcome을 모르더라도 학습이 가능하다.

<img src="https://dnddnjs.gitbooks.io/rl/content/TD6.png"/>

### Bias/Variance Trade-Off

<img src="https://dnddnjs.gitbooks.io/rl/content/main-qimg-01c15f01cf6a56c19313c2791d5a9ae1.jpg"/>

* Bias & Variance

  * 중앙으로부터 전체적으로 많이 벗어나게 되면 bias가 높아짐
  * 전체적으로 많이 퍼져있으면(중심으로부터 벗어난 것과는 관계 없이) variance가 높아짐

* 보통 두 개념은 서로 trade-off 관계에 있음

  * TD는 bias가, MC는 variance가 높음

* TD: 한 에피소드 안에서 계속 업데이트를 진행하게 되는데, 보통은 그 전의 상태가 그 후의 상태에 영향을 많이 주기 때문에 학습이 한 쪽으로 치우치게 된다.

  <img src="https://dnddnjs.gitbooks.io/rl/content/TD7.png"/>

* MC: 에피소드마다 학습을 진행하기 때문에 처음에 어떻게 했냐에 따라 전혀 다른 experience를 가질 수가 있다.



## 2) TD Control

### Sarsa

* TD(0)

  <img src="https://dnddnjs.gitbooks.io/rl/content/TD2.png"/>

  * Model-free control이 되기 위해서는 action-value function을 사용해야 한다.
    * 위의 TD(0) 식에서 value function을 action value function으로 바꿔주면 **Sarsa**

* Sarsa: 현재 state-action pair에서 다음 state와 action까지를 보고 update하는 기법

  <img src="https://dnddnjs.gitbooks.io/rl/content/TD8.png"/>

  * TD(0)를 action-value function으로 바꾸고, epsilon-greedypolicy improvement를 한 것.

    <img src="https://dnddnjs.gitbooks.io/rl/content/TD9.png"/>

  * on-policy TD control algorithm

    * 매 시각마다 현재의 Q값(어떤 state s에서 action a를 취할 경우 받을 return에 대한 기대값)를  imediate reward와 다음 action의 Q값을 가지고 update

    * policy는 따로 정의되지 않으며, Q값을 보고 epsilon-greedy하게 움직이는 것 자체가 policy가 된다.

      <img src="https://dnddnjs.gitbooks.io/rl/content/TD10.png"/>



### Eligibility Traces

* n-step TD,  TD(λ), eligibility trace는 최근 강화학습에서 많이 다루는 알고리즘은 아님

#### - TD prediction

1. n-step TD

   * Sarsa: update를 한번 할 때 하나의 정보밖에 알 수 없어 학습시 오래 걸리며 bias가 높음

     * TD와 MC의 장점을 다 살릴 수 없을까?

   * n-step TD: Update를 할 때 하나의 step만 보고 update를 하는 것이 아니라 n-step을 움직인 다음 update

     * 현재 시각 t에서 t+n까지 모았던 reward로 n-step return을 계산
     * 그 사이 방문했던 state들의 value function을 update
     * n이 terminal state이면 MC

     <img src="https://dnddnjs.gitbooks.io/rl/content/TD11.png"/>

     <img src="https://dnddnjs.gitbooks.io/rl/content/TD12.png"/>

2. Forward-View of TD(λ)

   * eg) Random Walker

     * C state에서 시작해 왼쪽/오른쪽으로 랜덤하게 움직이는 policy

     * true random function은 아래 그래프와 같으며, 이 value function을 experience를 통해 찾아내어야 한다.

       <img src="https://dnddnjs.gitbooks.io/rl/content/TD14.png"/>

     * 이 문제에 n-step TD prediction을 적용시킬 때, 적당한 n을 찾는 것이 쉽지 않음

       * x축은 α, y축은 true value function과의 에러일 때 α값에 따라 최적의 n값이 달라짐
       * 따라서 여러 n-step TD를 합할수 있다면 각 n의 장점을 다 취할 수 있을 것!

       <img src="https://dnddnjs.gitbooks.io/rl/content/TD13.png"/>

   * Forward-view TD(λ): n-step return에 대해 λ라는 weight를 이용해 geometrically weighted sum을 구하는 방법

     * 이렇게 하면 모든 n-step을 포함하며 총합은 1이 된다.

     * 이렇게 구한 λ-return을 원래 MC의 return 자리에 넣어주면 TD(λ)가 된다.

       <img src="https://dnddnjs.gitbooks.io/rl/content/TD15.png"/>

       <img src="https://dnddnjs.gitbooks.io/rl/content/TD16.png"/>

     

   * 하지만 모든 n-step을 포함해 계산하기 때문에 MC와 마찬가지로 episode가 끝나야 update를 진행할 수 있다.

     <img src="https://dnddnjs.gitbooks.io/rl/content/TD17.png"/>

3. Backward-View of TD(λ)

   <img src="https://dnddnjs.gitbooks.io/rl/content/TD18.png"/>

   * eligibility trace: 과거에 있었던 일 중 1) 얼마나 최근에 일어난 일이었는지, 2) 얼마나 자주 발생했었는지, 를 기준으로 과거의 일들을 기억해놓고 현재 받은 reward를 과거의 state들로 분배해준다.

   * TD(0)처럼 현재 update할 δ를 계산할 때, 현재의 value function으로만 update하는 것이 아님

   * 과거에 지나왔던 모든 state에 eligibility trace를 기억해두고, 그만큼 자신을 update

     * 현재의 경험을 통해 한번에 과거의 모든 state들의 value function을 update

     * 현재 경험이 과거 value function에 얼마나 영향을 주고 싶은지는 λ을 이용해 조절

     * 이러한 기법을 Backward -View TD(λ)라고 한다.

       <img src="https://dnddnjs.gitbooks.io/rl/content/TD19.png"/>

#### - TD Control

1. Sarsa(λ)

   * 마찬가지로 Sarsa에도 n-step Sars, forward-view Sarsa(λ), backward-view Sarsa(λ)가 있다.

     <img src="https://dnddnjs.gitbooks.io/rl/content/TD20.png"/>

     <img src="https://dnddnjs.gitbooks.io/rl/content/TD21.png"/>

     <img src="https://dnddnjs.gitbooks.io/rl/content/TD22.png"/>

   * backward-view pseudo code

     <img src="https://dnddnjs.gitbooks.io/rl/content/TD23.png"/>

   * Eligibility trace 예시

     <img src="https://dnddnjs.gitbooks.io/rl/content/TD24.png"/>