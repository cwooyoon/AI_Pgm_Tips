# Model_creation_basic.md

```
from keras.models import Sequential
from keras.layers import Dense

# models
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# compile
model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=['accuracy'])

# run
model.fit(X, Y, epochs=50, batch_size=10)
```

<img src="https://user-images.githubusercontent.com/54765256/90968930-50cb2100-e52d-11ea-8358-0dd00b323e31.png">

Optimizer

<img src="https://user-images.githubusercontent.com/54765256/90969001-29288880-e52e-11ea-94a8-b857a70e25ff.png">

<img src="https://user-images.githubusercontent.com/54765256/90969007-43626680-e52e-11ea-9202-eca7f974511d.png">

대표적 오차함수

<img src="https://user-images.githubusercontent.com/54765256/90970832-9a733600-e544-11ea-8f6f-164da1f9e35c.png">

1 epoch

학습 프로세스가 모든 샘플에 대해 한 번 실행되는 것

batch size

샘플을 한번에 몇개씩 처리할지 정함

너무 크면 학습 속도가 느려짐

너무 작으면 각 실행 값의 편차가 생겨서 전체 결과값이 불안정 해짐
