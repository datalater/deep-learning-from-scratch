import numpy as np

### 3.2.4 sigmoid

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


### 3.2.7 RelU

def relu(x):
    return np.maximum(0, x)


### 3.5.2 softmax

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # to prevent overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
