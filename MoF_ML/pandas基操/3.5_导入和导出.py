import pandas as pd

data = pd.read_excel('数据源.xls')
print(data)
# 自动加上索引

#      ID   NUM-1  NUM-2  NUM-3
# 0  36901    142    168    661
# 1  36902    147    520    131
# 2  36903    444    820    780
# 3  36904      5     85     47
# 4  36905    123    119    120

data.to_pickle('数据源.pickle')
