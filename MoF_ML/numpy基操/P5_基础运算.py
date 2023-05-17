import numpy as np

# a = np.array([10,20,30,40])
# b = np.arange(4)
# c = a
# print(c)
# # [10 20 30 40]
# d = a-b
# print(a,b,d)
# # [10 20 30 40] [0 1 2 3] [10 19 28 37]
# # 加减乘除都一样
#
# e = b**2
# print(e)
# # [0 1 4 9]
#
# f = 10*np.sin(a)    # cos tan。。。
# print(f)
# # [-5.44021111  9.12945251 -9.88031624  7.4511316 ]
#
# print(b)
# print(b < 3)
# # [0 1 2 3]
# # [ True  True  True False]

# 开始矩阵了
# a = np.array([[1,1],[0,1]])
# b = np.arange(4).reshape((2,2))
# print(a)
# # [[1 1]
# #  [0 1]]
# print(b)
# # [[0 1]
# #  [2 3]]
# print(b == 3)
# # [[False False]
# #  [False  True]]

# # 矩阵的乘法
# a = np.array([[1, 1], [0, 1]])
# b = np.arange(4).reshape((2, 2))
# print(a)
# print(b)
# c = a * b  # 逐个相乘
# c_dot = np.dot(a, b)  # 矩阵的乘法
# c_dot_2 = a.dot(b)  # 矩阵的乘法另一种表达形式
# print(c)
# # [[0 1]
# #  [0 3]]
# print(c_dot)
# # [[2 4]
# #  [2 3]]

# a = np.random.random((2,4))
a = np.array([[1, 2, 3], [0, 1,5]])
print(a)
# [[1 2 3]
#  [0 1 5]]

print(np.sum(a))
print(np.min(a))
print(np.max(a))
# 12
# 0
# 5

print(np.sum(a, axis=0))
# axis=1是对每一行进行求和；axis=0是对每一列求和
print(np.min(a,axis=1))
print(np.max(a,axis=1))
# [1 3 8]
# [1 0]
# [3 5]


