# import numpy as np

# A = np.array([1, 1, 1])
# B = np.array([2, 2, 2])
#
# print(np.vstack((A, B)))  # vertical stack
# # [[1 1 1]
# #  [2 2 2]]
#
# C = np.vstack((A, B))
# print(A.shape, C.shape)
# # (3,) (2, 3)
#
# D = np.hstack((A, B))  # horizontal stack
# print(D)
# # [1 1 1 2 2 2]
# print(A.shape, D.shape)
# # (3,) (6,)

import numpy as np
A = np.array([1, 1, 1])
B = np.array([2, 2, 2])
print(A.T)
# [1 1 1] #没有把横向的数列转成纵向的

print(A[np.newaxis, :])  # 在行向加了一个维度
print(A[np.newaxis, :].shape)
# [[1 1 1]]
# (1, 3)

print(A[:, np.newaxis])  # 在列向加了一个维度
print(A[:, np.newaxis].shape)
# [[1]
#  [1]
#  [1]]

# (3, 1)
C = np.concatenate((A,B,B,A),axis=0)
print(C)
# [1 1 1 2 2 2 2 2 2 1 1 1]

