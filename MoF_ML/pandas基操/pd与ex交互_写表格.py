import pandas as pd

basestation = "E:\code\python_code\python基础\莫烦\莫烦机器学习\pandas基操/数据源副.xls"
basestation_end = "E:\code\python_code\python基础\莫烦\莫烦机器学习\pandas基操/test_end.xls"
data = pd.read_excel(basestation)

data.to_excel(basestation_end)

# sheet_name，将数据存储在excel的那个sheet页面。
data.to_excel(basestation_end,sheet_name="sheet2")

# （3）na_rep，缺失值填充
data.to_excel(basestation_end,na_rep="NULL")

# (4) colums参数： sequence, optional，Columns to write 选择输出的的列。
data.to_excel(basestation_end,columns=["ID"])

# （5）header 参数： boolean or list of string，默认为True,可以用list命名列的名字。header = False 则不输出题头。
data.to_excel(basestation_end,header=["a","b","c","d"])
#     a   b   c   d
# 0   36901   142 168 661
# 1   36902   78  521 602
# 2   36903   144 600 521
# 3   36904   95  457 468
# 4   36905   69  596 695
# 5   36906   165 453
#
# （6）index : boolean, default True Write row names (index)
# 默认为True，显示index，当index=False 则不显示行索引（名字）。
# index_label : string or sequence, default None
# 设置索引列的列名。

data.to_excel(basestation_end,index=False)
# 输出：
# ID  NUM-1   NUM-2   NUM-3
# 36901   142 168 661
# 36902   78  521 602
# 36903   144 600 521
# 36904   95  457 468
# 36905   69  596 695
# 36906   165 453

data.to_excel(basestation_end,index_label=["f"])
# 输出：
# f   ID  NUM-1   NUM-2   NUM-3
# 0   36901   142 168 661
# 1   36902   78  521 602
# 2   36903   144 600 521
# 3   36904   95  457 468
# 4   36905   69  596 695
# 5   36906   165 453
