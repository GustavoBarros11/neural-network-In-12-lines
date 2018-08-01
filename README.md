# Neural Network in 12 lines
This code shows that creating a simple neural network can be really easy and straightfoward!
<br/><br/>
**SimplestEverNeuralNet.py**
```python
import numpy as np
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
w_0_1 = 2*np.random.random((3,4)) - 1
w_1_2 = 2*np.random.random((4,1)) - 1
for j in range(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,w_0_1))))
    l2 = 1/(1+np.exp(-(np.dot(l1,w_1_2))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(w_1_2.T) * (l1 * (1-l1))
    w_1_2 += l1.T.dot(l2_delta)
    w_0_1 += X.T.dot(l1_delta)
```
