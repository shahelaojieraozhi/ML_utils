import numpy as np
# a = np.arange(4)
# print(a)
# # [0 1 2 3]
# b = a
# print(b)
# # [0 1 2 3]
#
# d = b
#
# # a = b 则a和b完全是一个东西了，跟着变
# a[0] = 9
# print(a)
# # [9 1 2 3]
# print(b)
# # [9 1 2 3]
# print(b is a)
# # True
#
# print(d is a)
# # True
# d[1:3] = [22,33]
# print(d)
# # [ 9 22 33  3]
# print(a)
# # [ 9 22 33  3]

# 如果想把a的值赋给b，但是又不想关联起来：
a = np.arange(4)
print(a)
# [0 1 2 3]
b = a.copy()    # deep copy
print(b)
# [0 1 2 3]
a[3] = 45
print(b)
# [0 1 2 3]
print(a)
# [ 0  1  2 45]
