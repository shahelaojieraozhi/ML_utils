import pandas as pd
import numpy as np

# # merging two df by key(maybe used in database)
# left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
#                      'A': ['A0', 'A1', 'A2', 'A3'],
#                      'B': ['B0', 'B1', 'B2', 'B3']})
# right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
#                       'C': ['C0', 'C1', 'C2', 'C3'],
#                       'D': ['D0', 'D1', 'D2', 'D3']})
# print(left)
# #   key   A   B
# # 0  K0  A0  B0
# # 1  K1  A1  B1
# # 2  K2  A2  B2
# # 3  K3  A3  B3
# print(right)
# #   key   C   D
# # 0  K0  C0  D0
# # 1  K1  C1  D1
# # 2  K2  C2  D2
# # 3  K3  C3  D3
# res = pd.merge(left, right, on='key')
# print(res)
# #   key   A   B   C   D
# # 0  K0  A0  B0  C0  D0
# # 1  K1  A1  B1  C1  D1
# # 2  K2  A2  B2  C2  D2
# # 3  K3  A3  B3  C3  D3

# left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
#                      'key2': ['K0', 'K1', 'K0', 'K1'],
#                      'A': ['A0', 'A1', 'A2', 'A3'],
#                      'B': ['B0', 'B1', 'B2', 'B3']})
# right = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
#                      'key2': ['K0', 'K0', 'K0', 'K0'],
#                       'C': ['C0', 'C1', 'C2', 'C3'],
#                       'D': ['D0', 'D1', 'D2', 'D3']})
# print(left)
# print(right)
# res = pd.merge(left, right, on=['key1', 'key2'], how='inner')
# # how = ['left','right','outer','inner']
# print(res)
# #   key1 key2   A   B   C   D
# # 0   K0   K0  A0  B0  C0  D0
# # 1   K0   K0  A0  B0  C1  D1
# # 2   K1   K0  A2  B2  C2  D2

# df1 = pd.DataFrame({'col': [0, 1], 'col_left': ['a', 'b']})
# df2 = pd.DataFrame({'col': [1, 2, 2], 'col_right': [2, 2, 2]})
# print(df1)
# #    col col_left
# # 0    0        a
# # 1    1        b
# print(df2)
# #    col  col_right
# # 0    1          2
# # 1    2          2
# # 2    2          2
# res = pd.merge(df1, df2, on='col', how='outer', indicator=True)
# print(res)
# #    col col_left  col_right      _merge
# # 0    0        a        NaN   left_only
# # 1    1        b        2.0        both
# # 2    2      NaN        2.0  right_only
# # 3    2      NaN        2.0  right_only

# # 通过索引来合并
# left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
#                      'B': ['B0', 'B1', 'B2'],
#                      }, index=['K0', 'K1', 'K2'])
# right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
#                       'D': ['D0', 'D2', 'D3']},
#                      index=['K0', 'K2', 'K3'])
# print(left)
# #      A   B
# # K0  A0  B0
# # K1  A1  B1
# # K2  A2  B2
# print(right)
# #      C   D
# # K0  C0  D0
# # K2  C2  D2
# # K3  C3  D3
#
# res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
# print(res)
# #       A    B    C    D
# # K0   A0   B0   C0   D0
# # K1   A1   B1  NaN  NaN
# # K2   A2   B2   C2   D2
# # K3  NaN  NaN   C3   D3

# 处理overlapping
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
print(boys)
#     k  age
# 0  K0    1
# 1  K1    2
# 2  K2    3
girls = pd.DataFrame({'k': ['K0', 'K1', 'K3'], 'age': [4, 5, 6]})
print(girls)
#     k  age
# 0  K0    4
# 1  K1    5
# 2  K3    6
res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
print(res)
#     k  age_boy  age_girl
# 0  K0        1         4
# 1  K1        2         5
