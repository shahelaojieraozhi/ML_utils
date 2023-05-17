import numpy as np
A = np.arange(12).reshape((3,4))
print(A)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]


# # 实现纵向分割
# print(np.split(A,2,axis=1))
# # [array([[0, 1],
# #        [4, 5],
# #        [8, 9]]), array([[ 2,  3],
# #        [ 6,  7],
# #        [10, 11]])]
#
# # 实现横向分割，分成三块
# print(np.split(A, 3, axis=0))
# # [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]

# 如何实现不等量的分割：
print(np.array_split(A, 3, axis=1))
# [array([[0, 1],
#        [4, 5],
#        [8, 9]]), array([[ 2],
#        [ 6],
#        [10]]), array([[ 3],
#        [ 7],
#        [11]])]

print(np.vsplit(A, 3))
# [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
print(np.hsplit(A, 2))
# [array([[0, 1],
#        [4, 5],
#        [8, 9]]), array([[ 2,  3],
#        [ 6,  7],
#        [10, 11]])]
