from linear import *
from quadratic import *
from baye import *
from knn import *
from svr import *
from gbm import *
import matplotlib.pyplot as plt
import numpy as np


x=[]
y=[]

a,b = liner()
x.append(a)
y.append(b)

a,b = quad()
x.append(a)
y.append(b)

a,b = baye()
x.append(a)
y.append(b)

a,b = KNN()
x.append(a)
y.append(b)

a,b = SV()
x.append(a)
y.append(b)

a,b = GBM()
x.append(a)
y.append(b)

print(x)
print(y)

y_pos = np.arange(len(x))
plt.bar(y_pos, y, align='center', alpha=0.5)
plt.xticks(y_pos, x)
plt.ylabel('ROOT MEAN SQUARE ERROR')
plt.xlabel('MODELS')
plt.title('SOLAR ENERGY PREDICTIONS')

plt.show()
