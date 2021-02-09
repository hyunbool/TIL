# Neural Topic Model with Reinforcement Learning(2019)

# 1. Introduction

* Probabilistic topic model => NLP 분야에서 널리 사용되어져 옴.
  * 단어들이 단어 간 co-occurance pattern에 기반해 추론된 latent topic들로 부터 만들어진다고 가정
* Variational Autoencoder(VAE) => 깊고 복잡한 variance를 근사시키는데 효과적이라고 증명됨
  * 하지만 그동안의 VAE-based topic model들은 1) 단어와 latent topic 간 분포를 근사하거나 2) 학습된 토픽들의 질을 높이기 위해서가 아니라 학습된 latent topic vector를 이용해 원문을 reconstruct할 때 에러를 최소화 하기 위해 사용
  * topic coherence measure을 고려하지 않고 VAE-based topic model을 학습하게되면 topic에 대한 퀄리티를 조작하는데 어려움을 준다.
    * 이를 해결하기 위해서 learning object와 coherence score을 함께 고려하는 방법도 있지만, coherence score가 topic에 대한 unsupervised measure이기 때문에"best topic"에 대한 ground truth가 없어 실현 가능하지 않다.
  * 또한 현존하는 방법은 vocabulary size를 줄이기 위해 전처리 작업을 진행해야 한다.
    * word filtering은 대부분 heuristic하게 진행된다.
      * 자동으로 background word와 topic words를 구분해주는 시도도 있었지만, 이는 각 단어에 대해 background word인지 아닌지에 대한 switch variable을 정의해줘야 하기 때문에 모델의 계산량이 증가하거나 각 토픽 모델에 대해 각각 모델링을 진행해 주어야 하는 어려움이있다.
* 따라서 본 논문에서는 강화학습을 사용해
  *  topic coherence을 측정하는 부분을 neural topic model을 학습하는 부분과 통합하고 dynamic하게 background word를 필터링하는 새로운 프레임워크를 제시한다.
    * input document에 대해 우선 weight vector를 이용해 같은 constituent에 속하는 단어들은 높은 가중치(높은 coherence score)를 가지도록 constituent words를 샘플링한다.
      * 이때 constituent words들은 더 concentrated topic distribution을 가지게 된다.
    * 샘플링 된 단어들은 원 document를 reconstruct하기 위해 VAE-based neural topic model로 feed
      * reward signal이 각 단어에 대한 가중치 벡터를 업데이트하는데 사용된다.
        * 이렇게 되면 더이상 loss function에 coherence score을 더하지 않아도 된다.
* 실험 결과 20 Newsgroups와 NIPS data에 대해 topic coherence와 perplexity 면에서 기존의 프레임워크들을 능가하는 결과를 보였다.



## 2. Proposed Method

