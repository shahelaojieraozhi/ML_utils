import pandas as pd

basestation = r"E:\code\python_code\python基础\莫烦\莫烦机器学习\pandas基操/数据源.xls"
data = pd.read_excel(basestation)
print(data)
# #      ID   NUM-1  NUM-2  NUM-3
# # 0  36901    142    168    661
# # 1  36902    147    520    131
# # 2  36903    444    820    780
# # 3  36904      5     85     47
# # 4  36905    123    119    120

# 若sheetname=None是返回全表
data_1 = pd.read_excel(basestation, sheetname=[0, 1])  # 输出sheet1和sheet2
print(data_1)
print(type(data_1))
# OrderedDict([(0,      ID   NUM-1  NUM-2  NUM-3
# 0  36901    142    168    661
# 1  36902    147    520    131
# 2  36903    444    820    780
# 3  36904      5     85     47
# 4  36905    123    119    120), (1,      ID   NUM-1  NUM-2  NUM-3
# 0  36906    109    168     61
# 1  36907    439    520    131)])
# <class 'collections.OrderedDict'>

data = pd.read_excel(basestation, header=None)
print(data)
#        0      1      2      3
# 0    ID   NUM-1  NUM-2  NUM-3
# 1  36901    142    168    661
# 2  36902    147    520    131
# 3  36903    444    820    780
# 4  36904      5     85     47
# 5  36905    123    119    120

# header-取数起始行
data = pd.read_excel(basestation, header=[3])
print(data)
#    36903  444    820    780
# 0  36904      5     85     47
# 1  36905    123    119    120

# skiprows 参数：省略指定行数的数据
data = pd.read_excel(basestation, skiprows=[1])
print(data)
#      ID   NUM-1  NUM-2  NUM-3
# 0  36902    147    520    131
# 1  36903    444    820    780
# 2  36904      5     85     47
# 3  36905    123    119    120

# skip_footer参数：省略从尾部数的3行的数据
data = pd.read_excel(basestation, skip_footer=3)
print(data)
#      ID   NUM-1  NUM-2  NUM-3
# 0  36901    142    168    661
# 1  36902    147    520    131

# index_col参数：指定列为索引列，也可以使用u”strings”
data = pd.read_excel(basestation, index_col="NUM-3")
# print(data)
#          ID   NUM-1  NUM-2
# NUM-3
# 661    36901    142    168
# 131    36902    147    520
# 780    36903    444    820
# 47     36904      5     85
# 120    36905    123    119

# （7）names参数： 指定列的名字。
data = pd.read_excel(basestation, names=["a", "b", "c", "e"])
print(data)
#        a    b    c    e
# 0  36901  142  168  661
# 1  36902  147  520  131
# 2  36903  444  820  780
# 3  36904    5   85   47
# 4  36905  123  119  120

