# Chapter 3. Autoencoder

## Introduction

![img](https://blog.kakaocdn.net/dn/8JonH/btqFBec9cAF/mhxdDF930R0CrHs9NdUKv1/img.png)

* 보통 은닉층은 입력/출력층에 비해 작은 차원을 가진다!



![img](https://blog.kakaocdn.net/dn/OHinQ/btqFztcbowJ/Zu49rG8eWCK4kPJ9GYaKKK/img.png)

* 비지도 학습 방법인 차원 축소를 지도 학습(or self-learning)처럼 바꿔서 해결할 수 있기 때문에 주목을 받았음!

![Multi-Layer PerceptronLINEAR AUTOENCODER 3 / 24 𝐿(𝑥, 𝑦) 𝑥 ∈ ℝ 𝑑 𝑦 ∈ ℝ 𝑑 𝑧 ∈ ℝ 𝑑 𝑧 𝑧 = ℎ(𝑥) 𝑦 = 𝑔(ℎ 𝑥 ) ℎ(∙) 𝑔(∙) input out...](https://image.slidesharecdn.com/aes171113-180510014736/95/-50-1024.jpg?cb=1525916931)

* Linear AE의 경우 activation function 없이 사용
  * 이 경우 PCA와 같은 manifold를 학습하게 된다! 



## Stacking Autoencoder 

![Target 784 1000 1000 10 Input output Input 784 1000 784 W1 𝑥 ො𝑥 500 W1’ AutoencoderPRETRAINING 6 / 24 AUTOENOCDERS Stackin...](https://image.slidesharecdn.com/aes171113-180510014736/95/-53-1024.jpg?cb=1525916931)

![Target 784 1000 1000 500 10 Input output Input 784 1000 W1 1000 1000 fix 𝑥 𝑎1 ො𝑎1 W2 W2’ AutoencoderPRETRAINING 7 / 24 AUT...](https://image.slidesharecdn.com/aes171113-180510014736/95/-54-1024.jpg?cb=1525916931)

![Target 784 1000 1000 10 Input output Input 784 1000 W1 1000 fix 𝑥 𝑎1 ො𝑎2 W2fix 𝑎2 1000 W3 500 500 W3’ AutoencoderPRETRAINI...](https://image.slidesharecdn.com/aes171113-180510014736/95/-55-1024.jpg?cb=1525916931)

* 레이어 바이 레이어로 원본 데이터를 가장 잘 표현하는(원본 데이터로 잘 복원할 수 있는) weight를 학습해나간다.
* 그렇게 앞서 학습한 가중치는 고정시켜 두고 그 다음 레이어의 파라미터를 학습하게 된다.



![Target 784 1000 1000 10 Input output Input 784 1000 W1 1000 𝑥 W2 W3 500 500 10output W4  Random initialization Autoencode...](https://image.slidesharecdn.com/aes171113-180510014736/95/-56-1024.jpg?cb=1525916931)

* 그런 다음 Backpropagation을 이용해 fine-tune



## Denoising Autoencoder

![img](https://blog.kakaocdn.net/dn/ounJn/btqFzAI6pc7/2KCZVMxIADYpakuQZNmiK0/img.png)

* 입력시 노이즈를 추가해서 데이터를 입력한다. 



![img](https://blog.kakaocdn.net/dn/Bv4Pv/btqFA60Vtz7/lmuTYVuIlmCDqvCjePNxB0/img.png)

