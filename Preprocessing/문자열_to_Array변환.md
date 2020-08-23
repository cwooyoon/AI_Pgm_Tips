# 문자열 -> Array 

## sklearn

```
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../dataset/iris.csv',
                names=["sepal_length","sepal_width","petal_length", 
                      "petal_width", "species"])
                      
X = dataset[:,:4].astype(float)
Y_obj = dataset[:,4]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

Y_encoded = np_utils.to_categorical(Y)
Y_encoded

```

