# 폐암 수술환자 생존율 예측

## 웹 링크

https://m.blog.naver.com/ming_4u/221198595935

## 프로그램

```
import numpy
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import keras

# Versions
print(tf.__version__)
print(keras.__version__)

# 같은 결과 출력을 위한 seed 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 수술환자 데이터 로드
Data_set = numpy.loadtxt("../deeplearning/dataset/ThoraricSurgery.csv", delimiter=",")

# 환자기록 X, 결과 Y
X = Data_set[:, 0:17]
Y = Data_set[:, 17]

# models
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# compile
model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=['accuracy'])

# run
model.fit(X, Y, epochs=50, batch_size=10)

# result
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
```


