import pandas as pd
import numpy as np

# # concatenating
# df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
# print(df1)
# #      a    b    c    d
# # 0  0.0  0.0  0.0  0.0
# # 1  0.0  0.0  0.0  0.0
# # 2  0.0  0.0  0.0  0.0
# df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
# print(df2)
# #      a    b    c    d
# # 0  1.0  1.0  1.0  1.0
# # 1  1.0  1.0  1.0  1.0
# # 2  1.0  1.0  1.0  1.0
# df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])
# print(df3)
# #      a    b    c    d
# # 0  2.0  2.0  2.0  2.0
# # 1  2.0  2.0  2.0  2.0
# # 2  2.0  2.0  2.0  2.0
#
# res = pd.concat([df1, df2, df3], axis=0)
# # python中0就是横向的，1就是纵向的
# print(res)
# #      a    b    c    d
# # 0  0.0  0.0  0.0  0.0
# # 1  0.0  0.0  0.0  0.0
# # 2  0.0  0.0  0.0  0.0
# # 0  1.0  1.0  1.0  1.0
# # 1  1.0  1.0  1.0  1.0
# # 2  1.0  1.0  1.0  1.0
# # 0  2.0  2.0  2.0  2.0
# # 1  2.0  2.0  2.0  2.0
# # 2  2.0  2.0  2.0  2.0
#
# # 把索引变成连续的
# res = pd.concat([df1, df2, df3], axis=0 ,ignore_index=True)
# print(res)
# #      a    b    c    d
# # 0  0.0  0.0  0.0  0.0
# # 1  0.0  0.0  0.0  0.0
# # 2  0.0  0.0  0.0  0.0
# # 3  1.0  1.0  1.0  1.0
# # 4  1.0  1.0  1.0  1.0
# # 5  1.0  1.0  1.0  1.0
# # 6  2.0  2.0  2.0  2.0
# # 7  2.0  2.0  2.0  2.0
# # 8  2.0  2.0  2.0  2.0


# # join,['inner','outer']
# df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'],index=[1,2,3])
# print(df1)
# #      a    b    c    d
# # 1  0.0  0.0  0.0  0.0
# # 2  0.0  0.0  0.0  0.0
# # 3  0.0  0.0  0.0  0.0
# df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])
# print(df2)
# #     b    c    d    e
# # 2  1.0  1.0  1.0  1.0
# # 3  1.0  1.0  1.0  1.0
# # 4  1.0  1.0  1.0  1.0
#
# res = pd.concat([df1, df2])
# # 默认是：res = pd.concat([df1, df2]，jion='outer')
# print(res)
# #      a    b    c    d    e
# # 1  0.0  0.0  0.0  0.0  NaN
# # 2  0.0  0.0  0.0  0.0  NaN
# # 3  0.0  0.0  0.0  0.0  NaN
# # 2  NaN  1.0  1.0  1.0  1.0
# # 3  NaN  1.0  1.0  1.0  1.0
# # 4  NaN  1.0  1.0  1.0  1.0
#
# res = pd.concat([df1, df2],join='inner')
# print(res)
# #      b    c    d
# # 1  0.0  0.0  0.0
# # 2  0.0  0.0  0.0
# # 3  0.0  0.0  0.0
# # 2  1.0  1.0  1.0
# # 3  1.0  1.0  1.0
# # 4  1.0  1.0  1.0
#
# res = pd.concat([df1, df2], join='inner', ignore_index=True)
# print(res)
# #      b    c    d
# # 0  0.0  0.0  0.0
# # 1  0.0  0.0  0.0
# # 2  0.0  0.0  0.0
# # 3  1.0  1.0  1.0
# # 4  1.0  1.0  1.0
# # 5  1.0  1.0  1.0

# # join_axes
# res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])
# # axis=1是左右合并，索引对不上，所以要加个join_axes=[df1.index]
# print(res)
# #      a    b    c    d    b    c    d    e
# # 1  0.0  0.0  0.0  0.0  NaN  NaN  NaN  NaN
# # 2  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
# # 3  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
#
# # 对比一下：
# res = pd.concat([df1, df2], axis=1)
# print(res)
# #      a    b    c    d    b    c    d    e
# # 1  0.0  0.0  0.0  0.0  NaN  NaN  NaN  NaN
# # 2  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
# # 3  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
# # 4  NaN  NaN  NaN  NaN  1.0  1.0  1.0  1.0

# # append
# # 加了一行：
# df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
# print(df1)
# #      a    b    c    d
# # 0  0.0  0.0  0.0  0.0
# # 1  0.0  0.0  0.0  0.0
# # 2  0.0  0.0  0.0  0.0
# s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# res = df1.append(s1, ignore_index=True)
# print(res)
# #      a    b    c    d
# # 0  0.0  0.0  0.0  0.0
# # 1  0.0  0.0  0.0  0.0
# # 2  0.0  0.0  0.0  0.0
# # 3  1.0  2.0  3.0  4.0

# 三个相加
df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
print(df1)
#      a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
print(df2)
#      a    b    c    d
# 0  1.0  1.0  1.0  1.0
# 1  1.0  1.0  1.0  1.0
# 2  1.0  1.0  1.0  1.0

df3 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
res = df1.append([df2, df3], ignore_index=True)
print(res)
#      a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  1.0  1.0  1.0  1.0
# 4  1.0  1.0  1.0  1.0
# 5  1.0  1.0  1.0  1.0
# 6  1.0  1.0  1.0  1.0
# 7  1.0  1.0  1.0  1.0
# 8  1.0  1.0  1.0  1.0

