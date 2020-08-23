## v1

```
import tensorflow as tf

# seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)
```

## v2 - making
```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```
```
tf.compat.v1.set_random_seed(seed)
```
