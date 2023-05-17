import pandas as pd
import numpy as np

dates = pd.date_range('2021-09-22', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
print(df)
#              A   B   C   D
# 2021-09-22   0   1   2   3
# 2021-09-23   4   5   6   7
# 2021-09-24   8   9  10  11
# 2021-09-25  12  13  14  15
# 2021-09-26  16  17  18  19
# 2021-09-27  20  21  22  23
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan
print(df)
#              A     B     C   D
# 2021-09-22   0   NaN   2.0   3
# 2021-09-23   4   5.0   NaN   7
# 2021-09-24   8   9.0  10.0  11
# 2021-09-25  12  13.0  14.0  15
# 2021-09-26  16  17.0  18.0  19
# 2021-09-27  20  21.0  22.0  23

print(df.dropna(axis=0, how='any'))
# axis=0，则丢掉行
# any 任何里面有NAN，就把这行丢掉  all 只有在这行里面全是NAN的话才把它删掉
#              A     B     C   D
# 2021-09-24   8   9.0  10.0  11
# 2021-09-25  12  13.0  14.0  15
# 2021-09-26  16  17.0  18.0  19
# 2021-09-27  20  21.0  22.0  23

# 丢弃列
print(df.dropna(axis=1, how='any'))
#              A   D
# 2021-09-22   0   3
# 2021-09-23   4   7
# 2021-09-24   8  11
# 2021-09-25  12  15
# 2021-09-26  16  19
# 2021-09-27  20  23

# 把NAN填充为0
print(df.fillna(value=0))
#              A     B     C   D
# 2021-09-22   0   0.0   2.0   3
# 2021-09-23   4   5.0   0.0   7
# 2021-09-24   8   9.0  10.0  11
# 2021-09-25  12  13.0  14.0  15
# 2021-09-26  16  17.0  18.0  19
# 2021-09-27  20  21.0  22.0  23

# 找到缺失值，缺失的显示True
print(df.isnull())
#                 A      B      C      D
# 2021-09-22  False   True  False  False
# 2021-09-23  False  False   True  False
# 2021-09-24  False  False  False  False
# 2021-09-25  False  False  False  False
# 2021-09-26  False  False  False  False
# 2021-09-27  False  False  False  False

# 如果列表太大，不好直接找的话可以看看下面：
# 直接找到列表中是否有缺失值
print(np.any(df.isnull())==True)
# True


