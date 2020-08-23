# 다중분류 문제: 아이리스 품종 예측

출처: 모두의 딥러닝(조태호 저)

dataset/iris.csv

## 다중 분류 문제

아이리스는 꽃잎의 모양과 길이에 따라 여러 가지 품종으로 나뉨

사진을 보면 품종마다 비슷해 보이는데 과연 딥러닝을 사용하여 이들을 구별해 낼 수 있을까?

그림 12-1  아이리스의 품종

<img src="https://user-images.githubusercontent.com/54765256/90972422-cea22300-e553-11ea-8047-206d11ffef06.png">

아이리스 품종 예측 데이터는 책과 함께 제공하는 예제 파일의 dataset 폴더에서 찾을 수 있음(dataset/iris.csv)

표 12-1  아이리스 데이터의 샘플, 속성, 클래스 구분

<img src = "https://user-images.githubusercontent.com/54765256/90972436-f7c2b380-e553-11ea-858c-cb114bb282d7.png">

<img src = "https://user-images.githubusercontent.com/54765256/90972447-1163fb00-e554-11ea-9e11-7b5815d31865.png">

속성을 보니 우리가 앞서 다루었던 것과 중요한 차이는 바로 클래스가 2개가 아니라 3개임

즉, 참(1)과 거짓(0)으로 해결하는 것이 아니라, 여러개 중에 어떤 것이 답인지를 예측하는 문제임

다중 분류 (multi classification) :

     여러 개의 답 중 하나를 고르는 분류 문제
     
다중 분류 문제는 둘 중에 하나를 고르는 이항 분류(binary classification)와는 접근 방식이 조금 다름

```
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# 데이터 입력
df = pd.read_csv('../dataset/iris.csv', names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
```
<img src="https://user-images.githubusercontent.com/54765256/90972957-889b8e00-e558-11ea-9d61-47ccc80fa083.png">
```
# 그래프로 확인
sns.pairplot(df, hue='species')
plt.show()
```
<img src="https://user-images.githubusercontent.com/54765256/90973007-ee881580-e558-11ea-9d3c-c0726465df72.png">



















