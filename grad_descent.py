import numpy as np
import Constant


def training(X, Y, w):
    m = X.shape[1]  # example number
    J = 0
    grad_w = np.zeros((Constant.feature+1, 1))
    for i in range(0, Constant.iteration):
        w = w - Constant.learning_rate * grad_w
        output = np.dot(w.T, X)
        #  cost function
        J = 1 / (2 * m) * np.sum(np.square(output - Y), axis=1, keepdims=True)
        grad_w = 1 / m * np.dot((output - Y), X.T).T
        if i % 10 == 0:
            print("Cost at integration" + str(i) + ": " + str(J))

        # if i % 10 == 0:
        #     print("parametter at integration" + str(i) + ": " + str(w))

    return w
