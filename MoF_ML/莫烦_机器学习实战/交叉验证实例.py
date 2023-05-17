import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# 参考：https://www.cnblogs.com/volcao/p/9291831.html
digits = datasets.load_digits()
X = digits.data
y = digits.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=666)

''' 使用 train_test_split 并调参 '''
# best_score, best_p, best_k = 0, 0, 0
# for k in range(2, 10):
#     for p in range(1, 6):
#         knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=k, p=p)
#         knn_clf.fit(X_train, y_train)
#         score = knn_clf.score(X_test, y_test)
#         if score > best_score:
#             best_score, best_p, best_k = score, p, k
#
# print("Best K =", best_k)
# print("Best P =", best_p)
# print("Best score =", best_score)

'''
Best K = 3
Best P = 2
Best score = 0.9860917941585535
'''

'''使用交叉验证调参 '''
best_score, best_p, best_k = 0, 0, 0
for k in range(2, 10):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=k, p=p)
        scores = cross_val_score(knn_clf, X_train, y_train)
        score = np.mean(scores)
        if score > best_score:
            best_score, best_p, best_k = score, p, k

print("Best K =", best_k)
print("Best P =", best_p)
print("Best score =", best_score)

'''
4) 分析
    与 train_test_split 调参法的区别
    拟合方式：cross_val_score(knn_clf, X_train, y_train)，默认将 X_train 分割成 3 份，并得到 3 个模型的准确率；
            如果想将 X_train 分割成 k 份cross_val_score(knn_clf, X_train, y_train, cv=k)；
    判定条件：score = np.mean(scores)，交叉验证中取 3 个模型的准确率的平均值最高时对应的一组参数作为最终的最优超参数；

    与 train_test_split 方法相比，交叉验证过程中得到的最高准确率 score 较小；
    原因：在交叉验证中，通常不会过拟合某一组的测试数据，所以平均来讲所得准确率稍微低一些；

5) 其它
    交叉验证得到的最好的准确率（Best score = 0.9823599874006478），并不是最优模型相对于数据集的准确率；
    交叉验证的直接目的是获取最优的模型对应的超参数，而不是得到最优模型，当拿到最优的超参数后，就可以根据参数获取最佳的 kNN 模型；
'''

