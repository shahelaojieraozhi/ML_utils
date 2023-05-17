# 1、.loc与.iloc、.at与.iat区别结论：
# .loc与.iloc区别：.loc通过标签索引，不能使用位置索引；.iloc通过位置索引，不能通过标签索引
# .loc与.iloc相同：都可获取多行或多列或多行多列或单个值
# .at与.iat区别：同.loc与.iloc区别，.at通过标签获取单个值，.iat通过位置索引获取单个值
# .at与.iat相同：只能获取单个值，不能获取多个值。这也是与.loc和.iloc的区别
# 换句话说，.loc与.iloc函数功能包含.at与.iat的函数功能，.at与.iat访问数据的速度更快

from pandas import Series, DataFrame
import pandas as pd
import numpy as np

# loc
df1 = DataFrame(np.random.randint(0, 10, (4, 4)), index=['haha', 'dada', 'pipi', 'keke'], columns=['a', 'b', 'c', 'd'])
print(df1)
# 使用loc获取第二行第三列的值
print('使用loc获取第二行第三列的值:', df1.loc['dada', 'c'])
# 若使用位置索引，会报错，错误的演示：print(df1.loc[1,2])
# 输出：
#       a  b  c  d
# haha  5  7  1  0
# dada  2  9  5  3
# pipi  6  8  1  5
# keke  5  5  9  7
# 使用loc获取第二行第三列的值: 5

# 使用loc获取第三行的值
print(df1.loc['pipi'])
# 》结果
# a    6
# b    8
# c    1
# d    5
# Name: pipi, dtype: int32

# #使用loc获取第二三列的数据
print(df1.loc[:,['b','c']])
#       b  c
# haha  7  1
# dada  9  5
# pipi  8  1
# keke  5  9

# #使用loc获取第一二行及第三四列交叉的数据
print(df1.loc[['haha','pipi'],['c','d']])
#       c  d
# haha  1  0
# pipi  1  5


# ## iloc ###

# #使用iloc同理
# #iloc获取第二行的数据
print(df1.iloc[1])
# a    2
# b    9
# c    5
# d    3
# Name: dada, dtype: int32

# #iloc获取第二列的数据
print(df1.iloc[:,[1]])
#       b
# haha  7
# dada  9
# pipi  8
# keke  5

# ##.at及.iat
# .at获取第二行第三列的值
print(df1.at['dada', 'c'])
# 5

# .iat获取第二行第三列的值
print(df1.iat[1, 2])
# 5

# 使用get_value获取某个值,可以得到结果但是会告警，建议使用.at
print(df1.get_value('dada', 'c'))
# 5
