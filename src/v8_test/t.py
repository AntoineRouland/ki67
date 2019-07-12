import matplotlib.pyplot as plt
import numpy as np

a = [1 * i for i in range(0, 5)]
b = [2 * i for i in range(0, 5)]
c = [4 * i for i in range(0, 5)]
d = [a, b, c]

plt.xlim(10, 40)
plt.xlabel('Experts Values')
plt.ylabel('Method Outputs')

for i in range(len(d)):
    if i == 0:
        v = 30
    elif i == 1:
        v = 20
    elif i == 2:
        v = 15
    for j in range(len(d[i])):
        plt.plot(v, d[i][j], 'b+')
        plt.text(v+0.3, d[i][j], f'{j}')
    plt.plot(v, np.mean(d[i]), 'r+')
    p50 = np.percentile(d[i], 50)
    p75 = np.percentile(d[i], 75)
    plt.plot(v, p50, 'g+')
    plt.text(v+0.7, p50, '50th percentile')
    plt.plot(v, p75, 'g+')
    plt.text(v+0.7, p75, '75th percentile')


"""plt.plot([15] * len(a), a, 'b+')
plt.plot([30] * len(b), b, 'b+')
plt.plot([20] * len(c), c, 'b+')"""


plt.show()

