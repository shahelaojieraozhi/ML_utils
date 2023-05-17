from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
# from sklearn.datasets._samples_generator import make_classification
# 0.22版本已经弃用了，用上面那句
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# a = np.array([[10, 2.7, 3.6],
#               [-100, 5, -2],
#               [120, 20, 40]], dtype=np.float64)
# print(a)
# print(preprocessing.scale(a))
# # [[  10.     2.7    3.6]
# #  [-100.     5.    -2. ]
# #  [ 120.    20.    40. ]]
# # [[ 0.         -0.85170713 -0.55138018]
# #  [-1.22474487 -0.55187146 -0.852133  ]
# #  [ 1.22474487  1.40357859  1.40351318]]

# Normalization在skLearning里叫scale
# (X-mean)/std 计算时对每个属性/每列分别进行

# 换一种Normalization方式
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                           random_state=22, n_clusters_per_class=1, scale=100)
# 生成300个样本，2个特征值；两个比较相关的属性；
print(X)
#  [ 1.60272371e+02  6.60453240e+01]
#  [ 2.35217443e+02  5.91911131e+01]
#  .....300个二维特征值
print(y)
# [0 1 0 0 0 0 0 1 0 1 0 0 0 1 1 0 0 0 1 0 1 1 0 0 1 0 1 1 1 0 1 0 0 0 1 1 1
#  1 1 0 0 1 0 1 1 0 1 1 1 1 0 0 0 1 1 0 1 1 1 0 1 0 1 0 0 0 1 1 1 1 0 1 1 0
#  0 0 0 1 1 1 0 1 1 1 0 1 0 ....] 300个标签
plt.scatter(X[:, 0], X[:, 1], c='r')
plt.show()

# 使用preprocessing.minmax_scale 归一化：
X = preprocessing.minmax_scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))