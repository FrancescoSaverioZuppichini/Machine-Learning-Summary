import matplotlib.pyplot as plt
import math
import numpy as np

X = np.arange(-10.0, 10.0, 0.01)
print(X)
sig = lambda x: 1/(1 + math.e**(-x))

dsig = lambda x: sig(x) * (1 - sig(x))

MAX_ITER = 10

fig = plt.figure()

temp = sig(X)
plt.title('Sigmoid')

for i in range(MAX_ITER):

    plt.plot(temp, label="{}".format(i))

    temp = sig(temp)

plt.legend()
plt.show()
fig.savefig('./sigmoid_gradient_vanish.png')
fig.clear()

fig = plt.figure()

temp = X
plt.title('Derivative of Sigmoid')
for i in range(MAX_ITER):

    plt.plot(dsig(temp), label="{}".format(i))
    temp = sig(temp)

plt.legend()
plt.show()
fig.savefig('./sigmoid_gradient_vanish_derivative.png')
fig.clear()