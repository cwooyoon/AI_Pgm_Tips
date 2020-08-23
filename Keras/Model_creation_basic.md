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

<img src="https://user-images.githubusercontent.com/54765256/90969001-29288880-e52e-11ea-94a8-b857a70e25ff.png">

<img src="https://user-images.githubusercontent.com/54765256/90969007-43626680-e52e-11ea-9202-eca7f974511d.png">

대표적 오차함수

<img src="https://user-images.githubusercontent.com/54765256/90970832-9a733600-e544-11ea-8f6f-164da1f9e35c.png">

