# Chap 2. Manifold Learning

## Introduction

![img](https://blog.kakaocdn.net/dn/rQO7z/btqFANNpZhU/SnKg5GtojDAtcadcruNgpk/img.png)

* 고차원 데이터가 있다고 하자.
* 샘플들을 잘 아우르는 subspace(manifold)를 잘 찾으면 데이터의 차원을 잘 축소시킬 수 있다.
* What is it useful for?
  * Data compression
  * Data visualization
  * Curse of dimensionality
  * Discovering most important features



## Curse of Dimensionality

![img](https://blog.kakaocdn.net/dn/d5MlXG/btqFAgvN4rC/HWdJXFzUmDS6q9r6KT3SMk/img.png)

* 데이터 차원이 증가할수록 밀도는 급속도로 희박해지게 된다.

![img](https://blog.kakaocdn.net/dn/crSf1C/btqFz1Td2OK/5QkPVQJG01KCox3PxTI3mk/img.png)

* 그렇기 때문에 데이터가 잘 밀집되어 있는 공간(manifold)를 찾아보자!



![img](https://blog.kakaocdn.net/dn/cJHVPO/btqFzzv7ZxP/U3e3mKvjPb76bzeiB5MYqK/img.png)



![img](https://blog.kakaocdn.net/dn/bCrTgM/btqFzAPg7zy/oKrkpapSbSVBzGiyuokGY1/img.png)

* 같은 measure이라도 어떤 공간에서 측정하냐에 따라서 그 거리가 달라진다.



![img](https://blog.kakaocdn.net/dn/6JHKu/btqFANNrqz8/AqgGlPlXBkHLteLkfkZNlK/img.png)

* 다양한 차원 축소 방법들이 존재하며, AE는 Non-Linear 방법론 중 하나!