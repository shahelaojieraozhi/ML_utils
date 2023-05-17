import numpy as np

# A = np.arange(3, 15)
# print(A)
# print(A[3])  # 找A中索引为3的对应的元素
# # [ 3  4  5  6  7  8  9 10 11 12 13 14]
# # 6

# A = np.arange(3, 15).reshape((3, 4))
# print(A)
# # [[ 3  4  5  6]
# #  [ 7  8  9 10]
# #  [11 12 13 14]]
#
# print(A[2])  #
# # [11 12 13 14]
#
# print(A[1][1])
# # 8
# print(A[2][1])
# # 12
# print(A[2, 1])  # 同上
# # 12
#
# print(A[0, :])  # 第一行的所有数
# # [3 4 5 6]
#
# print(A[:, 0])  # 第一列行的所有数
#
# print(A)
# print(A[0, 1:3])  # 第一行的第二个到第三个
# # [4 5]

# A = np.arange(3, 15).reshape((3, 4))
# print(A)
# [[ 3  4  5  6]
#  [ 7  8  9 10]
#  [11 12 13 14]]

# for row in A:
#     print(row)
#     # [3 4 5 6]
#     # [ 7  8  9 10]
#     # [11 12 13 14]
#
# for column in A.T:
#     print(column)
#     # [3  7 11]
#     # [4  8 12]
#     # [5  9 13]
#     # [6 10 14]


A = np.arange(3, 15).reshape((3, 4))
print(A)

print(A.flatten())
# <numpy.flatiter object at 0x00000254706A2540>
# [ 3  4  5  6  7  8  9 10 11 12 13 14]

for item in A.flat:
    print(item)
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 11
# 12
# 13
# 14
