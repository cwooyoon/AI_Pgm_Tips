# 경사하강법

출처: 모두의딥러닝 2판(조태호 저)

## 오차 수정하기: 경사 하강법

경사 하강법( gradient descent) :
     그래프에서 오차를 비교하여 가장 작은 방향으로 이동시키는 방법이 있는데 바로 미분 기울기를 이용

그림 4-1  기울기 a와 오차와의 관계: 적절한 기울기를 찾았을 때 오차가 최소화된다.

<img src="https://user-images.githubusercontent.com/54765256/91016985-97ea0c80-e628-11ea-8e35-194dcdf38ba6.png">

그림 4-2  순간 기울기가 0인 점이 곧 우리가 찾는 최솟값 m이다.

<img src="https://user-images.githubusercontent.com/54765256/91017162-cd8ef580-e628-11ea-8be7-765b1e738801.png">

1 |     에서 미분을 구함
2 | 구해진 기울기의 반대 방향(기울기가 +면 음의 방향, -면 양의 방향)으로 얼마간 이동시킨      에서 미분을 구함(그림 4-3 참조).
3 | 위에서 구한 미분 값이 0이 아니면 위 과정을 반복함

그림 4-3  최솟점 m을 찾아가는 과정

<img src="https://user-images.githubusercontent.com/54765256/91017267-f7481c80-e628-11ea-833d-aa24568298bd.png">

그림 4-4  학습률을 너무 크게 잡으면 한 점으로 수렴하지 않고 발산한다.

<img src="https://user-images.githubusercontent.com/54765256/91017341-1646ae80-e629-11ea-877e-948e12f52636.png">

## 코딩으로 확인하는 경사 하강법

최솟값을 구하기 위해서는 이차 함수에서 미분을 해야 함

그 이차 함수는 평균 제곱 오차를 통해 나온다는 것임

평균 제곱 오차의 식을 다시 옮겨 보면 다음과 같음

<img src="https://user-images.githubusercontent.com/54765256/91017743-99680480-e629-11ea-8a01-d881e073fb7e.png">

<img src="https://user-images.githubusercontent.com/54765256/91017829-b270b580-e629-11ea-8a63-da75b3298ba5.png">

<img src="https://user-images.githubusercontent.com/54765256/91017878-c4eaef00-e629-11ea-81de-b0a447635dec.png"?

<img src="https://user-images.githubusercontent.com/54765256/91017935-d7652880-e629-11ea-85f4-09e7cbe76866.png">

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#공부시간 X와 성적 Y의 리스트를 만듭니다.
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

#그래프로 나타내 봅니다.
plt.figure(figsize=(8,5))
plt.scatter(x, y)
plt.show()
```
<img src="https://user-images.githubusercontent.com/54765256/91018042-0380a980-e62a-11ea-817b-0764920ce7ca.png">
```
#리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸어 줍니다.(인덱스를 주어 하나씩 불러와 계산이 가능해 지도록 하기 위함입니다.)
x_data = np.array(x)
y_data = np.array(y)

# 기울기 a와 절편 b의 값을 초기화 합니다.
a = 0
b = 0

#학습률을 정합니다.
lr = 0.03 

#몇 번 반복될지를 설정합니다.
epochs = 2001 

#경사 하강법을 시작합니다.
for i in range(epochs): # epoch 수 만큼 반복
    y_hat = a * x_data + b  #y를 구하는 식을 세웁니다
    error = y_data - y_hat  #오차를 구하는 식입니다.
    a_diff = -(2/len(x_data)) * sum(x_data * (error)) # 오차함수를 a로 미분한 값입니다. 
    b_diff = -(2/len(x_data)) * sum(error)  # 오차함수를 b로 미분한 값입니다. 
    a = a - lr * a_diff  # 학습률을 곱해 기존의 a값을 업데이트합니다.
    b = b - lr * b_diff  # 학습률을 곱해 기존의 b값을 업데이트합니다.
    if i % 100 == 0:    # 100번 반복될 때마다 현재의 a값, b값을 출력합니다.
        print("epoch=%.f, 기울기=%.04f, 절편=%.04f" % (i, a, b))
```
```
epoch=0, 기울기=27.8400, 절편=5.4300
epoch=100, 기울기=7.0739, 절편=50.5117
epoch=200, 기울기=4.0960, 절편=68.2822
epoch=300, 기울기=2.9757, 절편=74.9678
epoch=400, 기울기=2.5542, 절편=77.4830
epoch=500, 기울기=2.3956, 절편=78.4293
epoch=600, 기울기=2.3360, 절편=78.7853
epoch=700, 기울기=2.3135, 절편=78.9192
epoch=800, 기울기=2.3051, 절편=78.9696
epoch=900, 기울기=2.3019, 절편=78.9886
epoch=1000, 기울기=2.3007, 절편=78.9957
epoch=1100, 기울기=2.3003, 절편=78.9984
epoch=1200, 기울기=2.3001, 절편=78.9994
epoch=1300, 기울기=2.3000, 절편=78.9998
epoch=1400, 기울기=2.3000, 절편=78.9999
epoch=1500, 기울기=2.3000, 절편=79.0000
epoch=1600, 기울기=2.3000, 절편=79.0000
epoch=1700, 기울기=2.3000, 절편=79.0000
epoch=1800, 기울기=2.3000, 절편=79.0000
epoch=1900, 기울기=2.3000, 절편=79.0000
epoch=2000, 기울기=2.3000, 절편=79.0000
```
```
# 앞서 구한 기울기와 절편을 이용해 그래프를 그려 봅니다.
y_pred = a * x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()
```
<img src="https://user-images.githubusercontent.com/54765256/91018269-55c1ca80-e62a-11ea-8b60-1b754d17a630.png">










