# Wine : 베스트 모델 만들기

출처: 모두의 딥러닝(조태호 저)

dataset/wine.csv

Redwine 1,599, whitewine 4898개

포르투갈 서북쪽의 대서양을 맞닿고 위치한 비뉴 베르드(Vinho Verde) 지방에서 만들어진 와인을 측정한 데이터임

레드와인 샘플 1,599개를 등급과 맛, 산도를 측정해 분석하고 화이트와인 샘플4,898개를 마찬가지로 분석해 데이터를 만듦

## 데이터의 확인과 실행

먼저 df_pre라는 공간에 데이터를 불러옴

그런 다음 sample( ) 함수를 사용하여 원본 데이터의 몇 %를 사용할지를 지정함

sample( ) 함수는 원본 데이터에서 정해진 비율만큼 랜덤으로 뽑아오는 함수임

frac = 1이라고 지정하면 원본 데이터의 100%를 불러오라는 의미임

frac = 0.5로 지정하면 50%만 랜덤으로 불러옴

```
# 데이터 입력
df_pre = pd.read_csv('../dataset/wine.csv', header=None)
df = df_pre.sample(frac=1)
```

<img src="https://user-images.githubusercontent.com/54765256/90976050-d1614000-e574-11ea-880b-40ed1f1e00ed.png">

<img src = "https://user-images.githubusercontent.com/54765256/90976066-feadee00-e574-11ea-9e3f-716475981c8f.png">

총 6497개의 샘플이 있음을 알 수 있음

13개의 속성이 각각 무엇인지는 데이터를 내려받은 UCI 머신러닝 저장소에서 확인할 수 있음

<img src="https://user-images.githubusercontent.com/54765256/90976081-1a18f900-e575-11ea-90af-3688ad74fd59.png">

```
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# 데이터 입력
df_pre = pd.read_csv('../dataset/wine.csv', header=None)
df = df_pre.sample(frac=1)

df.head()

dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

# 모델 설정
model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#모델 컴파일
model.compile(loss='binary_crossentropy',
           optimizer='adam',
           metrics=['accuracy'])

# 모델 실행
model.fit(X, Y, epochs=200, batch_size=200)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
```

## 모델 업데이트하기

모델을 그냥 저장하는 것이 아니라 에포크(epoch)마다 모델의 정확도를 함께 기록하면서 저장해 보자

```
from keras.callbacks import ModelCheckpoint
import os

# 모델 저장 폴더 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
   os.mkdir(MODEL_DIR)

# 모델 저장 조건 설정
modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 모델 실행 및 저장
model.fit(X, Y, validation_split=0.2, epochs=200, batch_size=200, verbose=0, callbacks=[checkpointer],
        save_best_only=True)
```

## 그래프로 확인하기

딥러닝 프레임워크가 만들어 낸 모델을 업데이트 하는 과정임

이를 위해서는 에포크를 얼마나 지정할지를 결정해야 함

학습을 반복하는 횟수가 너무 적어도 안 되고 또 너무 많아도 과적합을 일으키므로 문제가 있음

모델의 학습 시간에 따른 정확도와 테스트 결과를 그래프를 통해 확인해 보자

다음으로 그래프로 표현하기 위한 라이브러리를 불러오고 오차와 정확도의 값을정함

y_vloss에 테스트셋(33%)으로 실험한 결과의 오차 값을 저장함

y_acc에 학습셋(67%)으로 측정한 정확도의 값을 저장함

```
# 모델 저장 폴더 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
   os.mkdir(MODEL_DIR)

# 모델 저장 조건 설정
modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 모델 실행 및 저장
history = model.fit(X, Y, validation_split=0.33, epochs=3500, batch_size=500)

# y_vloss에 테스트셋으로 실험 결과의 오차 값을 저장
y_vloss=history.history['val_loss']

# y_acc 에 학습 셋으로 측정한 정확도의 값을 저장
y_acc=history.history['accuracy']

# x값을 지정하고 정확도를 파란색으로, 오차를 빨간색으로 표시
x_len = numpy.arange(len(y_acc))
plt.plot(x_len, y_vloss, "o", c="red", markersize=3)
plt.plot(x_len, y_acc, "o", c="blue", markersize=3)

plt.show()
```

그림 14-1  학습 진행에 따른 학습셋과 테스트셋의 정확도 그래프

<img src="https://user-images.githubusercontent.com/54765256/90976519-d0caa880-e578-11ea-8b53-50b48776220d.png">

## 학습의 자동 중단

학습이 진행될수록 학습셋의 정확도는 올라가지만 과적합 때문에 테스트셋의 실험 결과는 점점 나빠지게 됨

EarlyStopping( ) 함수 :

     케라스에는 이렇게 학습이 진행되어도 테스트셋 오차가 줄지 않으면 학습을 멈추게 하는 함수

```
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import pandas as pd
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

df_pre = pd.read_csv('../dataset/wine.csv', header=None)
df = df_pre.sample(frac=0.15)

dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
           optimizer='adam',
           metrics=['accuracy'])

# 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)

# 모델 실행
model.fit(X, Y, validation_split=0.2, epochs=2000, batch_size=500, callbacks=[early_stopping_callback])

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
```











