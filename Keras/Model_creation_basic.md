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
