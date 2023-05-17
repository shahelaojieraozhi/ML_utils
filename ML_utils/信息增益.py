"""
python3
-- coding: utf-8 --
-------------------------------
@Author : RAO ZHI
@Email : raozhi@mails.cust.edu.cn
-------------------------------
@File : 信息增益.py
@Software : PyCharm
@Time : 2023/5/12 10:37
-------------------------------
"""

from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)
print(mutual_info_classif(x, y))


