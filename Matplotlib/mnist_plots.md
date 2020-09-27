# MNIST Plots

## scikit-learn MNIST

* Init
```
import warnings
warnings.filterwarnings(action='ignore') 

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
```

* Data load
```
import numpy as np  # 넘파이 패키지 임포트
import matplotlib.pylab as plt  # 맷플롯립 패키지 임포트

from sklearn.datasets import load_iris  # 사이킷런 패키지 임포트

iris = load_iris()  # 데이터 로드
iris.data[0, :]  # 첫 번째 꽃의 데이터

iris.feature_names

iris.target_names
iris.target
iris.data.shape
iris.data.ndim
```

* MNIST Sampls plotting
```
from sklearn.datasets import load_digits  # 패키지 임포트

digits = load_digits()  # 데이터 로드
samples = [0, 10, 20, 30, 1, 11, 21, 31]  # 선택된 이미지 번호
d = []
for i in range(8):
    d.append(digits.images[samples[i]])

plt.figure(figsize=(8, 2))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(d[i], interpolation='nearest', cmap=plt.cm.bone_r)
    plt.grid(False); plt.xticks([]); plt.yticks([])
    plt.title("image {}".format(i + 1))
plt.suptitle("숫자 0과 1 이미지")
plt.tight_layout()
plt.show()
```

* Results
<img src="https://user-images.githubusercontent.com/54765256/94352679-2d067800-00a3-11eb-8af1-78e3ea4bb11e.png">

* 벡터화된 이미지
```
v = []
for i in range(8):
    v.append(d[i].reshape(64, 1))  # 벡터화
    
plt.figure(figsize=(8, 3))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(v[i], aspect=0.4,
               interpolation='nearest', cmap=plt.cm.bone_r)
    plt.grid(False); plt.xticks([]); plt.yticks([])
    plt.title("벡터 {}".format(i + 1))
plt.suptitle("벡터화된 이미지", y=1.05)
plt.tight_layout(w_pad=7)
plt.show()
```

* Result

<img src="https://user-images.githubusercontent.com/54765256/94352693-6dfe8c80-00a3-11eb-9718-9a256e4a700a.png">






