import numpy as np
from plot import plot_data
import Constant
from grad_descent import training
import matplotlib.pyplot as plt

my_data = np.genfromtxt('models/time_series_covid19_confirmed_global.csv', delimiter=',', dtype=None, encoding="utf8")
Y = my_data[140][4:].astype(int)
# print(Y)
date_axis = my_data[0][4:]
example_number = my_data[140][4:].shape[0]
print("Dates count:" + str(example_number))

feature = Constant.feature
X = np.zeros((feature, example_number))
x_index = np.arange(0, example_number)
# initialize example data
for i in range(0, feature):
    X[i, :] = np.power(x_index, i + 1)

# normalize X:
distance = X[:, example_number - 1]
X = (X - np.sum(X, axis=1, keepdims=True) / example_number)
for i in range(0, feature):
    X[i, :] = X[i, :] / distance[i]

# initialize parameters
w = np.zeros((feature + 1, 1))

X_train = np.insert(X, 0, np.ones((1, example_number)), axis=0)

w_optimize = training(X_train, Y, w)

print(w_optimize)

output = np.dot(w_optimize.T, X_train)

# generate data for next week
next_week = example_number + 7
X_next_week = np.zeros((feature, 7))
x_index_next = np.arange(example_number, next_week)
for i in range(0, feature):
    X_next_week[i, :] = np.power(x_index_next, i + 1)
# normalize X_next:
X_next_week = (X_next_week - np.sum(X, axis=1, keepdims=True) / example_number)
for i in range(0, feature):
    X_next_week[i, :] = X_next_week[i, :] / distance[i]

X_next_week = np.insert(X_next_week, 0, np.ones((1, 7)), axis=0)

predict = np.dot(w_optimize.T, np.concatenate((X_train, X_next_week), axis=1))

# print(X)
# print(X_next_week)
x_index_next = np.arange(0, next_week)


plt.plot(x_index_next, predict.T, 'g--')
plt.ylabel('Infected cases in Japan')
plt.xlabel('Date')
plt.show()

plt.plot(x_index, Y, 'ro', x_index, output.T, 'b-')
plt.ylabel('Infected cases in Japan')
plt.xlabel('Date')
plt.show()


