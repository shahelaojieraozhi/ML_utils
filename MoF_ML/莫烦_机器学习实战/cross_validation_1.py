import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# sklearn.cross_validation在1.9版本以后就被弃用了，1.9版本的以后的小伙伴可以用sklearn.model_selection就行了
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data   # 特征向量
y = iris.target     # 种类标签

# 把train_data和test_data分开：
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
# print(X_test)
# print(y_train)      # 可以看出标签已经被打乱了
# [1 0 2 0 1 0 2 0 0 1 1 2 0 1 2 2 1 1 0 1 2 1 0 1 0 1 2 1 2 1 0 2 2 0 1 2 0
#  2 1 2 1 0 2 1 2 0 2 1 2 1 2 1 1 2 1 1 2 1 1 0 2 0 1 0 1 1 1 1 0 2 2 1 1 1
#  0 0 2 2 0 0 0 2 0 0 2 2 1 0 0 0 2 1 0 0 2 1 2 0 0 2 1 1 1 2 2 1 2 1 1 2 2
#  2]

# knn = KNeighborsClassifier(n_neighbors=5)       # n_neighbors=5，考虑数据周围5个点进行训练
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print(y_pred)  # 显示全部X_test的预测结果
# # [2 0 2 2 2 1 2 0 0 2 0 0 0 1 2 0 1 0 0 2 0 2 1 0 0 0 0 0 0 2 1 0 2 0 1 2 2 1]
# print(knn.score(X_test, y_test))

# # 把data分成五组不同的training and test再进行训练并求平均值
# knn = KNeighborsClassifier(n_neighbors=5)
# scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
# # 把data分成五组不同的training and test
# print(scores)
# # [0.96666667 1 0.93333333 0.96666667 1]
#
# # 平均一下：
# print(scores.mean())
# # 0.9733333333333334

# 测试多少邻值效果最好：
k_range = range(1, 50)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')       # for classification
    # loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error')     # for regression
    k_scores.append(scores.mean())
print(k_scores)

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
# 由图可以看出：
# k值最好选择： 12——20左右

