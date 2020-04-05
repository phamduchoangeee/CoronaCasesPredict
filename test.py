import numpy as np

a = np.arange(9).reshape((3, 3))
b = np.arange(9).reshape((3, 3))
# b = [[1],
#      [2],
#      [3]]
# a_x = a[:,1]
# print(a/a_x)
c= [a,b]
print(np.sum(a, axis=1, keepdims=True) )
