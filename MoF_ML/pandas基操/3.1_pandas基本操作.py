import pandas as pd
import numpy as np
# s = pd.Series([1,3,6,np.nan,44,1])
# print(s)
# # 0     1.0
# # 1     3.0
# # 2     6.0
# # 3     NaN
# # 4    44.0
# # 5     1.0
# # dtype: float64

# dates = pd.date_range('2021-08-17', periods=6)
# print(dates)
#
# df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
# print(df)
# #                    a         b         c         d
# # 2021-08-17 -0.034046 -0.247738 -0.024117 -1.791032
# # 2021-08-18  0.097339 -1.069289  0.259134  0.357519
# # 2021-08-19  0.865789 -0.047085  1.364423 -0.934774
# # 2021-08-20  0.439360  1.302942 -0.136647 -0.462389
# # 2021-08-21  0.771514  1.717387 -0.952004 -1.562190
# # 2021-08-22 -1.212993  0.210176 -1.198396  1.604202
#
# df1 = pd.DataFrame(np.arange(12).reshape((3, 4)))
# print(df1)
# #    0  1   2   3
# # 0  0  1   2   3
# # 1  4  5   6   7
# # 2  8  9  10  11

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20210817'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'fpp'})
print(df2)
#      A          B    C  D      E    F
# 0  1.0 2021-08-17  1.0  3   test  fpp
# 1  1.0 2021-08-17  1.0  3  train  fpp
# 2  1.0 2021-08-17  1.0  3   test  fpp
# 3  1.0 2021-08-17  1.0  3  train  fpp
print(df2.dtypes)
# A           float64
# B    datetime64[ns]
# C           float32
# D             int32
# E          category
# F            object
# dtype: object
print(df2.index)
# Int64Index([0, 1, 2, 3], dtype='int64')
print(df2.columns)
# Index(['A', 'B', 'C', 'D', 'E', 'F'], dtype='object')
print(df2.values)
# [[1.0 Timestamp('2021-08-17 00:00:00') 1.0 3 'test' 'fpp']
#  [1.0 Timestamp('2021-08-17 00:00:00') 1.0 3 'train' 'fpp']
#  [1.0 Timestamp('2021-08-17 00:00:00') 1.0 3 'test' 'fpp']
#  [1.0 Timestamp('2021-08-17 00:00:00') 1.0 3 'train' 'fpp']]

# 描述这个DataFrame 忽略日期(B)和EF,只处理数字
print(df2.describe())
#          A    C    D
# count  4.0  4.0  4.0
# mean   1.0  1.0  3.0
# std    0.0  0.0  0.0
# min    1.0  1.0  3.0
# 25%    1.0  1.0  3.0
# 50%    1.0  1.0  3.0
# 75%    1.0  1.0  3.0
# max    1.0  1.0  3.0

# 转置
print(df2.T)
#                      0         ...                             3
# A                    1         ...                             1
# B  2021-08-17 00:00:00         ...           2021-08-17 00:00:00
# C                    1         ...                             1
# D                    3         ...                             3
# E                 test         ...                         train
# F                  fpp         ...                           fpp
#
# [6 rows x 4 columns]

# axis=0 表示对行（上下）进行操作计算，axis=1表示对列（左右）进行操作计算。可以把 1 看作一竖表示列，很自然。
print(df2.sort_index(axis=1,ascending=False))
#      F      E  D    C          B    A
# 0  fpp   test  3  1.0 2021-08-17  1.0
# 1  fpp  train  3  1.0 2021-08-17  1.0
# 2  fpp   test  3  1.0 2021-08-17  1.0
# 3  fpp  train  3  1.0 2021-08-17  1.0

print(df2.sort_index(axis=0,ascending=False))
#      A          B    C  D      E    F
# 3  1.0 2021-08-17  1.0  3  train  fpp
# 2  1.0 2021-08-17  1.0  3   test  fpp
# 1  1.0 2021-08-17  1.0  3  train  fpp
# 0  1.0 2021-08-17  1.0  3   test  fpp

print(df2.sort_values(by='E'))
#      A          B    C  D      E    F
# 0  1.0 2021-08-17  1.0  3   test  fpp
# 2  1.0 2021-08-17  1.0  3   test  fpp
# 1  1.0 2021-08-17  1.0  3  train  fpp
# 3  1.0 2021-08-17  1.0  3  train  fpp
