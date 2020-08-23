# Various data loading methods

## using numpy

```
Data_set = numpy.loadtxt("../deeplearning/dataset/ThoraricSurgery.csv", delimiter=",")
```
Slicing...

```
# 환자기록 X, 결과 Y
X = Data_set[:, 0:17]
Y = Data_set[:, 17]
```
