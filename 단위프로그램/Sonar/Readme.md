# Sonar data

출처: 모두의 딥러닝 (조태호 저)

1988년 존스홉킨스대학교의 세즈노프스키(Sejnowski) 교수는 2년 전 힌튼 
      교수가 발표한 역전파 알고리즘에 관심을 가지고 있었음
      
그는 은닉층과 역전파가 얼마나 큰 효과가 있는지를 직접 실험해 보고 싶었음

광석과 일반 돌을 가져다 놓고 음파 탐지기를 쏜 후 그 결과를 데이터로 정리함

오차 역전파 알고리즘을 사용한 신경망이 과연 얼마나 광석과 돌을 구분하는 데 효과적인지 알아보기 위해서임

## Code

```
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# 데이터 입력
df = pd.read_csv('../dataset/sonar.csv', header=None)
'''
# 데이터 개괄 보기
print(df.info())

# 데이터의 일부분 미리 보기
print(df.head())
'''
dataset = df.values
X = dataset[:,0:60]
Y_obj = dataset[:,60]

# 문자열 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 모델 설정
model = Sequential()
model.add(Dense(24,  input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy'])


# 모델 실행
model.fit(X, Y, epochs=200, batch_size=5)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
```
## 과적합 이해하기

완전히 새로운 데이터에 적용하면 이 선을 통해 정확히 두 그룹으로 나누지 못하게 됨

딥러닝 같은 알고리즘을 충분히 조절하여 가장 나은 모델이 만들어지면, 이를 실생활에 대입하여 활용하는 것이 바로 머신러닝의 개발 순서

그림 13-2  학습셋과 테스트셋

<img src="https://user-images.githubusercontent.com/54765256/90975347-52690900-e56e-11ea-8fb5-11fd62766208.png">

학습 데이터를 이용해 정확도를 측정한 것은 데이터에 들어있는 모든 샘플을 그대로 테스트에 활용한 결과임

학습에 사용된 샘플은 테스트에 쓸 수 없으므로 학습 단계에서 테스트할 샘플은 자동으로 빼고,
이를 테스트한 결과를 모아 정확도를 계산하는 것

이러한 방법은 빠른 시간에 모델 성능을 파악하고 수정할 수 있도록 도와 줌

머신러닝의 최종 목적은 과거의 데이터를 토대로 새로운 데이터를 예측하는 것

테스트셋을 만들어 정확한 평가를 병행하는 것이 매우 중요함

학습셋만 가지고 평가할때, 층을 더하거나 에포크(epoch) 값을 높여 실행 횟수를 늘리면 정확도가 계속해서 올라갈 수 있음

학습이 깊어져서 학습셋 내부에서의 성공률은 높아져도 테스트셋에서는 효과가 없다면 과적합이 일어나고 있는 것

그림 13-3  학습이 계속되면 학습셋에서의 정확도는 계속 올라가지만, 테스트셋에서는 과적합이 발생

<img src="https://user-images.githubusercontent.com/54765256/90975377-83493e00-e56e-11ea-94f9-84cb9abcd097.png">

그림 13-4  학습셋과 테스트셋 정확도 측정의 예(RP Gorman et.al.,1998)

<img src="https://user-images.githubusercontent.com/54765256/90975395-bb508100-e56e-11ea-9eb2-135dce5d68df.png">

```
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

df = pd.read_csv('../dataset/sonar.csv', header=None)

print(df.info())
print(df.head())

dataset = df.values
X = dataset[:,0:60]
Y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 학습 셋과 테스트 셋의 구분
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(24,  input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=130, batch_size=5)

# 테스트셋에 모델 적용
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))


```

## k겹 교차 검증

딥러닝 혹은 머신러닝 작업을 할 때 늘 어려운 문제 중 하나는 알고리즘을 충분히 
     테스트하였어도 데이터가 충분치 않으면 좋은 결과를 내기가 어렵다는 것
     
이러한 단점을 보완하고자 만든 방법이 바로 k겹 교차 검증(k-fold cross validation)

k겹 교차 검증 :

     데이터셋을 여러 개로 나누어 하나씩 테스트셋으로 사용하고 나머지를 모두 합해서 학습셋으로 사용하는 방법, 
이렇게 하면 가지고 있는 데이터의 100%를 테스트셋으로 사용할 수 있음

그림 13-5  5겹 교차 검증의 도식

<img src="https://user-images.githubusercontent.com/54765256/90975671-5d716880-e571-11ea-9861-a86ce9439757.png">
```
from sklearn.model_selection import StratifiedKFold

# 10개의 파일로 쪼갬
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

# 빈 accuracy 배열
accuracy = []

# 모델의 설정, 컴파일, 실행
for train, test in skf.split(X, Y):
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X[train], Y[train], epochs=100, batch_size=5)
    k_accuracy = "%.4f" % (model.evaluate(X[test], Y[test])[1])
    accuracy.append(k_accuracy)

# 결과 출력
print("\n %.f fold accuracy:" % n_fold, accuracy)
```



