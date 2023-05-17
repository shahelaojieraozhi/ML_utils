import numpy as np

# A = np.arange(2, 14).reshape(3, 4)
# print(A)
# print(np.argmin(A))
# # 0 最小值的索引
# print(np.argmax(A))
# # 11 最大值的索引
# print(np.mean(A))  # 求平均值
# print(A.mean())  # 同上功能
#
# #   同上
# print(np.average(A))  # 求平均值
# # print(A.average())  # 这个是错误的;
#
# print(np.median(A))  # 求中位数

# A = np.arange(2, 14).reshape(3, 4)
# print(A)
# print(np.cumsum(A))  # 累加处理
# # [[ 2  3  4  5]
# #  [ 6  7  8  9]
# #  [10 11 12 13]]
#
# # [ 2  5  9 14 20 27 35 44 54 65 77 90]
#
# print(np.diff(A))  # 累差处理
# # [[ 2  3  4  5]
# #  [ 6  7  8  9]
# #  [10 11 12 13]]
#
# # [[1 1 1]
# #  [1 1 1]
# #  [1 1 1]]
#
# print(np.nonzero(A))  # 找非零元素的横纵坐标
# #   (array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int64),
# #   array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
# # 第一个array是非零元素的横坐标，第二个是纵坐标

# A = np.arange(14, 2, -1).reshape(3, 4)
# print(A)
# print(np.sort(A))  # 排序
# # [[14 13 12 11]
# #  [10  9  8  7]
# #  [ 6  5  4  3]]
#
# # [[11 12 13 14]
# #  [ 7  8  9 10]
# #  [ 3  4  5  6]]
#
# print(A)
# print(np.transpose(A))  # 转置
# # [[14 13 12 11]
# #  [10  9  8  7]
# #  [ 6  5  4  3]]
# # [[14 10  6]
# #  [13  9  5]
# #  [12  8  4]
# #  [11  7  3]]
#
# print(A.T)  # 同上
# # [[14 10  6]
# #  [13  9  5]
# #  [12  8  4]
# #  [11  7  3]]
#
# print((A.T).dot(A))  # A矩阵乘A的转置
# # [[332 302 272 242]
# # [302 275 248 221]
# # [272 248 224 200]
# # [242 221 200 179]]

A = np.arange(14, 2, -1).reshape(3, 4)
print(A)
print(np.clip(A, 5, 9))  # 把比五小和比五大的数变为9和5
# [[9 9 9 9]
#  [9 9 8 7]
#  [6 5 5 5]]

print(np.mean(A, axis=0))  # 对列求平均值
print(np.mean(A, axis=1))  # 对行求平均值
