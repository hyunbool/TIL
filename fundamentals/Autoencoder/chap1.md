# Chap 1. Revisit Deep Neural Networks

* 목표: DNN 학습에 대한 해석 시 ML Density Estimation에 대한 이해



## ML Problem

![img](https://blog.kakaocdn.net/dn/PLOuX/btqFyQSJrjm/ABsDCJjhIKnFkaJWEIAG8K/img.png)



* 과정
  * Collect Traning Data
  * Define functions
  * Learning/Training
  * Predicting/Testing

![img](https://blog.kakaocdn.net/dn/MAEfr/btqFzHtPiuG/1zQHF5bzLzjKq1UKvt3DPK/img.png)

* Define function <= 이 부분에서 모델로 DNN을 사용하게 되는 것
* 학습해야 할 파라미터: weight & bias
* Loss 측정: MLE, Cross Entropy
  * 아무 Loss Function이나 쓰지 못한다!
  * 제약조건: Back Propagation
    * 가정1: Training Data 전체 loss function은 각 loss에 대한 합
    * 가정2: Loss function의 input은 네트워크 출력 값과 타켓 값

![img](https://blog.kakaocdn.net/dn/bTLsq7/btqFz2RZeJy/IWdmXbyKUgkf1dQW0iSef0/img.png)

* Learning/Training: Loss를 최소화하는 θ를 찾는다! => Gradient Descent
  * Iterative하게 파라미터를 업데이트
    * θ에 대한 업데이트: loss가 줄어드는 방향으로 이동
    * 언제 stop할 것인가?: loss가 변하지 않는 경우 stop

![img](https://blog.kakaocdn.net/dn/Xz75I/btqFzAPbuQd/yVdpZt09LG43KJcLC4MSsK/img.png)

* 그럼 어떻게 θ를 업데이트?
  * Taylor Expansion: 모든 미분 term을 사용하는 것이 아니라 1차 미분까지만 사용
    * 더 많은 차수의 미분값을 사용할 수록 error가 작아지긴 함!
  * Approximation
    * Loss function의 차이값은 <img src="https://latex.codecogs.com/gif.latex?%5Cbigtriangledown%20L%20%5Ccdot%20%5Cbigtriangleup%20%5Ctheta"/>
    * <img src="https://latex.codecogs.com/gif.latex?%5Cbigtriangleup%20%5Ctheta"/> 에 음수값을 취하면 loss function의 값이 항상 음수가 된다!



![Deep Neural Networks 5 / 17 01. Collect training data 02. Define functions 03. Learning/Training 04. Predicting/Testing DB...](https://image.slidesharecdn.com/aes171113-180510014736/95/-13-1024.jpg?cb=1525916931)

* 3번째 줄: 샘플별 평균 gradient
* 4번째 줄: 그런데 샘플 개수가 많아지면 저 값도 너무 커짐! 따라서 SGD 사용해 M개에 대한 평균 gradient를 구함



![img](https://blog.kakaocdn.net/dn/PT3cl/btqFz2Et8bT/TeOXBWdJnGInpHcnpq3Cok/img.png)

* Backpropagation
  * 출력단에서부터 error signal를 구한다.
  * 각 레이어별 error signal를 순차적으로(역순으로) 구한다.
  * 각 레이어별 구한 error signal에 대한 bias 미분값
  * 각 레이어별 구한 error signal에 대한 weight 미분값



## Loss Function

### View 1. Back Propagation 

#### 1. MSE

![View-Point I : BackpropagationLOSS FUNCTION 7 / 17 ∑ w b Input : 1.0 Output : 0.0 w=-1.28 b=-0.98 a=+0.09 w =-0.68 b =-0.6...](https://image.slidesharecdn.com/aes171113-180510014736/95/-15-1024.jpg?cb=1525916931)

* weight와 bias에 대한 초기값
  * sigmoid의 미분값이 0에 가까운지 여부
    * update 후의 값도 0에 가깝기 때문에 update가 잘 일어나지 않는다.
    * 거기에 레이어 앞단으로 갈 수록 값이 계속 작아짐
      * Gradient Vanishing 문제



#### 2. Cross Entropy

![img](https://blog.kakaocdn.net/dn/bneGBr/btqFAqx5QW2/m1PnkKt37y23LDvV4xGR7K/img.png)

* Error Signal을 구할 때 sigmoid 미분 term이 포함되지 않기 때문에 초기값에 민감하지 않게 update가 진행될 수 있다.



### View 2. Maximum Likelihood

![img](https://blog.kakaocdn.net/dn/SxjkD/btqFA575iCz/nV3RKGsJYMTSJwUXkzndi0/img.png)

* 네트워크 출력값에 대한 해석 = 네트워크의 출력값이 주어질 때, 우리가 원하는 정답이 나올 확률이 높길(같길) 바라게 된다!
  * conditional probability model을 가우시안으로 할 때
    * 네트워크의 출력은 확률분포를 정의하기 위한 파라미터를 추정하는 것
    * 가우시안 => 정의하기 위해서는 평균이 필요한데
      * 네트워크 출력값을 평균으로 해석했을 때 이 평균값이 y(정답)이 같아지는 것을 원한다 => Maximum Likelihood의 관점

* 이렇게 확률적으로 해석하게 되면 샘플링을 할 수 있다는 장점이 있다!



![View-Point II : Maximum Likelihood 12 / 17 01. Collect training data 02. Define functions 03. Learning/Training 04. Predic...](https://image.slidesharecdn.com/aes171113-180510014736/95/-20-1024.jpg?cb=1525916931)

* NNLL이 앞서 말한 loss function의 두가지 조건을 충족시키는지를 확인해보자.
  * i.i.d condition
    * 각 conditional probability의 곱으로 표현할 수 있으니 독립
    * 각 샘플별 확률분포가 모두 같다.

![img](https://blog.kakaocdn.net/dn/bZZZmw/btqFz16CKuM/8heAstPUccUdIIZJT5Xlk0/img.png)

* 가우시안 분포를 따르게 되면 loss function = MSE와 똑같아진다!
* 베르누이 분포 => CE와 동일한 수식이 된다!

![View-Point II : Maximum Likelihood 15 / 17 Gaussian distribution Categorical distribution 𝑓𝜃 𝑥𝑖 = 𝜇𝑖 𝑓𝜃 𝑥𝑖 = 𝑝𝑖 Distributi...](https://image.slidesharecdn.com/aes171113-180510014736/95/-23-1024.jpg?cb=1525916931)



![img](https://blog.kakaocdn.net/dn/clThSd/btqFzcaagtT/8Jxbl25tslqo7Gm2iwaD2k/img.png)



![View-Point II : Maximum Likelihood 17 / 17 Connection to Autoencoders Autoencoder LOSS FUNCTION REVISIT DNN Variational Au...](https://image.slidesharecdn.com/aes171113-180510014736/95/-25-1024.jpg?cb=1525916931)

