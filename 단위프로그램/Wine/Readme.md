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























