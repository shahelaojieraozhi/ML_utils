import pandas as pd
import numpy as np

dates = pd.date_range('2021-09-22', periods=6)
# print(dates)

df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
print(df)
#              A   B   C   D
# 2021-09-22   0   1   2   3
# 2021-09-23   4   5   6   7
# 2021-09-24   8   9  10  11
# 2021-09-25  12  13  14  15
# 2021-09-26  16  17  18  19
# 2021-09-27  20  21  22  23

# # 简单方法求列和行
# print(df['A'])
# # 2021-09-22     0
# # 2021-09-23     4
# # 2021-09-24     8
# # 2021-09-25    12
# # 2021-09-26    16
# # 2021-09-27    20
# # Freq: D, Name: A, dtype: int32
# print(df.A)
# # 2021-09-22     0
# # 2021-09-23     4
# # 2021-09-24     8
# # 2021-09-25    12
# # 2021-09-26    16
# # 2021-09-27    20
# # Freq: D, Name: A, dtype: int32
# 这两种选数的结果是一样的

# print(df[0:3])
# #             A  B   C   D
# # 2021-09-22  0  1   2   3
# # 2021-09-23  4  5   6   7
# # 2021-09-24  8  9  10  11

print(df['2021-09-22':'2021-09-25'])
#              A   B   C   D
# 2021-09-22   0   1   2   3
# 2021-09-23   4   5   6   7
# 2021-09-24   8   9  10  11
# 2021-09-25  12  13  14  15

# 高级一点的方法求列和行
# 见loc和iloc
