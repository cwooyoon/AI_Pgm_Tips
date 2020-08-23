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

그래프를 보니, 사진으로 볼 때는 비슷해 보이던 꽃잎과 꽃받침의 크기와 너비가 품종별로 차이가 있음을 알 수 있음

속성별로 어떤 연관이 있는지를 보여 주는 상관도 그래프를 통해 프로젝트의 감을 잡고 프로그램 전략을 세울 수 있음

이제 케라스를 이용해 아이리스의 품종을 예측해 보자

Iris-setosa, Iris-virginica 등 데이터 안에 문자열이 포함되어 있음
                                    
numpy보다는 pandas로 데이터를 불러와 X와 Y값을 구분하는 것이 좋음
```
# 데이터 분류
dataset = df.values
X = dataset[:,0:4].astype(float)
Y_obj = dataset[:,4]

# 문자열을 숫자로 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = np_utils.to_categorical(Y)
```
또한, Y값이 이번에는 숫자가 아니라 문자열임

문자열을 숫자로 바꿔 주려면 클래스 이름을 숫자 형태로 바꿔 주어야 함

이를 가능하게 하는 함수가 sklearn 라이브러리의 LabelEncoder( ) 함수임

## 원-핫 인코딩

array(['Iris-setosa', 'Iris-versicolor','Iris-virginica'])가 array([1,2,3])로 바뀜

활성화 함수를 적용하려면 Y 값이 숫자 0과 1로 이루어져 있어야 함

이 조건을 만족시키려면 tf.keras.utils.categorical( ) 함수를 적용해야 함

이에 따라 Y 값의 형태는 다음과 같이 변형됨

## Model
```
# 모델의 설정
model = Sequential()
model.add(Dense(16,  input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

# 모델 실행
model.fit(X, Y_encoded, epochs=50, batch_size=1)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))
```










