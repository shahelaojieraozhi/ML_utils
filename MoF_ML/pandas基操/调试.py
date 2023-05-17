import pandas as pd
import numpy as np
import datetime
import tqdm


def process(rawData):
    print(list(rawData))
    i = 0
    while i < len(rawData):
        rawData.loc[i, 'GPS (m/s)'] = rawData.iat[i, 2] / 3.6
        i += 1
    i = 1
    while i < len(rawData):
        timediff = rawData.iat[i, 0] - rawData.iat[i - 1, 0]
        speeddiff = rawData.iat[i, 3] - rawData.iat[i - 1, 3]
        rawData.loc[i, ' (m/s^2)'] = speeddiff / timediff
        i += 1
    print(rawData.head(10))
    rawData.to_excel("E:\code\python_code\python基础\莫烦\莫烦机器学习\pandas基操\questionD.xls")


basestation = "E:\code\python_code\python基础\莫烦\莫烦机器学习\pandas基操\文件1缩减版.xls"
df = pd.read_excel(basestation)
# print(df)
# #                           时间  GPS车速  X轴加速度  ...      空燃比  发动机负荷百分比  进气流量
# # 0   2017/12/18 13:42:13.000.    0.0  0.000  ...   0.1465        22  2.30
# # 1   2017/12/18 13:42:14.000.    0.0  0.000  ...   0.1465        21  2.39

# print(list(df))
# ['时间', 'GPS车速', 'X轴加速度', 'Y轴加速度', 'Z轴加速度', '经度', '纬度', '发动机转速', '扭矩百分比', '瞬时油耗', '油门踏板开度', '空燃比', '发动机负荷百分比', '进气流量']

# print(df['时间'])
# # # 0     2017/12/18 13:42:13.000.
# # # 1     2017/12/18 13:42:14.000.
# # # 2     2017/12/18 13:42:15.000.
# # # .....

# 把时间换格式
df['时间'] = pd.to_datetime(df['时间'], format='%Y/%m/%d %H:%M:%S.000.')
# print(df['时间'])
# 0    2017-12-18 13:42:13
# 1    2017-12-18 13:42:14
# 2    2017-12-18 13:42:15
# ...

# now = datetime.datetime.now()
# print(now)
# # 2021-09-08 10:50:23.334241
# print(now.strftime('%a, %b %d %H:%M'))
# # Wed, Sep 08 10:50

begin = datetime.datetime.strptime('2017-11-01 00:00:00', '%Y-%m-%d %H:%M:%S')
# print(begin)
# 2017-11-01 00:00:00


df['unix 时间'] = df['时间'].map(lambda x:(x-begin).total_seconds())
# print(df)
df.to_excel("E:\code\python_code\python基础\莫烦\莫烦机器学习\pandas基操\调试.xls")

col = list(df)
# print(col)

rawData = df[col[1:-3]]
# print(rawData)
#     GPS车速  X轴加速度  Y轴加速度  Z轴加速度   ...    扭矩百分比   瞬时油耗  油门踏板开度     空燃比
# 0     0.0  0.000 -0.396 -0.900   ...       18  58.02   0.000  0.1465
# 1     0.0  0.000 -0.378 -0.882   ...       17  60.30   0.000  0.1465
# 2     0.0  0.000 -0.396 -0.882   ...       17  55.24   0.000  0.1464
# rawData.to_excel("E:\code\python_code\python基础\莫烦\莫烦机器学习\pandas基操\调试1.xls")


df['时间'] = df['时间'].astype('str')
# print(df['时间'])
# 0     2017-12-18 13:42:13
# 1     2017-12-18 13:42:14
# 2     2017-12-18 13:42:15
# ....

process(df)

# i=0
# while i<len(df):
# if df.iat[i, 9] == 0 or df.iat[i, 10] == 0 or df.iat[i, 3] > 120:
# df.drop(df.index[i], inplace=True)
# else:
# i += 1
# df.to_excel('/Users/yangkai/Downloads/questionD/ 3 .xlsx')
# df[' ']=pd.to_datetime(df[' '],format='%Y-%m-%d %H:%M:%S')
# df['DIFF']=df[' '].diff(1).dt.seconds
# df['DIFF'].fillna(0)
# print(df.head(10))
# print(len(df))
# print(list(df))
# i️dleSpeedprocess(df)
# i️dleSpeed(df)
# Interpolation(df)
# removeError(df)
# removeAbnormalAcceleration(df)



