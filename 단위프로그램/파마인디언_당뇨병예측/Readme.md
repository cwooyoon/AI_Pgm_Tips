# 파마인디언_당뇨병예측

출처: 모두의딥러닝(조태호 저)

dataset/pima-indians-diabetes.csv

## 데이터 정제(Preprocessing)의 중요성

데이터의 양보다 훨씬 중요한 것은, ‘필요한’ 데이터가 얼마나 많은가임

준비된 데이터가 우리가 사용하려는 머신러닝과 딥러닝에 얼마나 효율적으로사용되게끔 가공됐는지가 역시 중요함

머신러닝 프로젝트의 성공과 실패는 얼마나 좋은 데이터를 가지고 시작하느냐에 영향을 많이 받음

여기서 좋은 데이터란 내가 알아내고자 하는 정보를 잘 담고 있는 데이터를 말함

한쪽으로 치우치지 않고, 불필요한 정보를 가지고 있지 않으며, 왜곡되지 않은 데이터여야 함

머신러닝, 딥러닝 개발자들은 데이터를 들여다 보고 분석할 수 있어야 함

내가 이루고 싶은 목적에 맞춰 가능한 한 많은 정보를 모았다면 이를 머신러닝과 딥러닝에서 사용할 수 있게 잘 정제된 데이터 형식으로 바꿔야 함

이 작업은 모든 머신러닝 프로젝트의 첫 단추이자 가장 중요한 작업
 
UCI 데이터 저장소 참조

http://archive.ics.uci.edu

## 피마 인디언 데이터 분석하기

비만이 유전 및 환경, 모두의 탓이라는 것을 증명하는 좋은 사례가 바로 미국 남서부에 살고 있는 피마 인디언의 사례

피마 인디언 데이터의 샘플, 속성, 클래스 구분(8개의 속성, 2개 클래스)

<img src="https://user-images.githubusercontent.com/54765256/90971001-82041b00-e546-11ea-87a9-4b41eae49e0f.png">

<img src="https://user-images.githubusercontent.com/54765256/90971022-b7106d80-e546-11ea-9349-cea9574b7a71.png">

## pandas를 활용한 데이터 조사

데이터를 다룰 때에는 데이터를 다루기 위해 만들어진 라이브러리를 사용하는 것이 좋음

파이썬 데이터 관련 라이브러리 중 pandas를 사용해 데이터를 불러와 보겠음(run_project/02_Pima_Indian.ipynb)

```
# pandas 라이브러리를 불러옵니다.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 피마 인디언 당뇨병 데이터셋을 불러옵니다. 불러올 때 각 컬럼에 해당하는 이름을 지정합니다.
df = pd.read_csv('../dataset/pima-indians-diabetes.csv',
               names = ["pregnant", "plasma", "pressure", "thickness", "insulin", 
               "BMI", "pedigree", "age", "class"])
```

Csv :

     comma separated values file의 약자로, 콤마(,)로 구분된 데이터들의 모음이란 뜻
     
헤더(header)  :

     csv 파일에는 데이터를 설명하는 한 줄이 파일 맨 처음에 나옴
     
우리가 가진 csv 파일에는 헤더가 없음

이에 names라는 함수를 통해 속성별 키워드를 지정해 줌

이제 불러온 데이터의 내용을 간단히 확인하고자 head( ) 함수를 이용하여 데이터의 첫 다섯 줄을 불러옴

```
# 처음 5줄을 봅니다.
print(df.head(5))

# 데이터의 전반적인 정보를 확인해 봅니다.
print(df.info())

# 각 정보별 특징을 좀더 자세히 출력합니다.
print(df.describe())

# 데이터 중 임신 정보와 클래스 만을 출력해 봅니다.
print(df[['plasma', 'class']])
```





