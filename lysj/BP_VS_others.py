from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
import datetime
from pandas.plotting import radviz
import argparse, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import joblib
from itertools import cycle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import joblib
from itertools import cycle

dataset = pd.read_excel('E:\\File_cache\\feiq\\Recv Files\\data_9_120.xlsx')
data = dataset.iloc[:, :9].values
label = dataset.iloc[:, -2].values

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(data)
y = label
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

from sklearn.model_selection import cross_val_score

'search for an optimal value of hidden numbers for BP model'
# hidden_range = range(1, 10)
# bp_scores = []
# for i in hidden_range:
#     # rfc = RandomForestClassifier(n_estimators=i)
#     svc = SVC(C=1, kernel='rbf', degree=3, gamma='scale', random_state=1)
#     scores = cross_val_score(svc, X_train, y_train, cv=10, scoring='accuracy')
#     bp_scores.append(scores.mean())
#
# print(bp_scores)  # [xx, xxx, .... ]
# print(max(bp_scores))
# # 保存一下
# pd.DataFrame(np.array(bp_scores)).to_csv('./output/十折交叉验证结果_SVM_筛选n_estimators.csv')
#
# plt.plot(hidden_range, bp_scores)
# plt.xlabel("Value of h for HiddenNum")
# plt.ylabel("Cross validated accuracy")
#
# plt.show()


# rfc = RandomForestClassifier(n_estimators=14)
# rfc.fit(X_train, y_train)
# y_predict_bp = rfc.predict(X_test)
# print("准确率：", accuracy_score(y_test, y_predict_bp))  # 其实是调参后

# clf = DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# y_predict_bp = clf.predict(X_test)
# print("准确率：", accuracy_score(y_test, y_predict_bp))  # 其实是调参后

# hidden_range = range(1, 31)
# bp_scores = []
# for i in hidden_range:
#     # rfc = RandomForestClassifier(n_estimators=i)
#     svc = SVC(C=i, kernel='rbf', degree=3, gamma='scale', random_state=1)
#     scores = cross_val_score(svc, X_train, y_train, cv=10, scoring='accuracy')
#     bp_scores.append(scores.mean())
#
# print(bp_scores)  # [xx, xxx, .... ]
# print(max(bp_scores))
# # 保存一下
# pd.DataFrame(np.array(bp_scores)).to_csv('./output/十折交叉验证结果_SVM_筛选C.csv')
#
# plt.plot(hidden_range, bp_scores)
# plt.xlabel("Value of h for HiddenNum")
# plt.ylabel("Cross validated accuracy")
#
# plt.show()


# svc = SVC(C=1, kernel='rbf', degree=3, gamma='scale', random_state=1)
#
# svc.fit(X_train, y_train)
# y_predict_bp = svc.predict(X_test)
# print("准确率：", accuracy_score(y_test, y_predict_bp))  # 其实是调参后


# hidden_range = range(1, 31)
# bp_scores = []
# for i in hidden_range:
#     # rfc = RandomForestClassifier(n_estimators=i)
#     knn = KNeighborsClassifier(n_neighbors=i)
#     scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
#     bp_scores.append(scores.mean())
#
# print(bp_scores)  # [xx, xxx, .... ]
# print(max(bp_scores))
# # 保存一下
# pd.DataFrame(np.array(bp_scores)).to_csv('./output/十折交叉验证结果_KNN_筛选n_neighbors.csv')
#
# plt.plot(hidden_range, bp_scores)
# plt.xlabel("Value of h for HiddenNum")
# plt.ylabel("Cross validated accuracy")

# plt.show()

# knn = KNeighborsClassifier(n_neighbors=2)

# knn.fit(X_train, y_train)
# y_predict_bp = knn.predict(X_test)
# print("准确率：", accuracy_score(y_test, y_predict_bp))  # 其实是调参后


hidden_range = range(1, 31)
bp_scores = []
for i in hidden_range:
    clf = DecisionTreeClassifier(max_depth=i)
    # rfc = RandomForestClassifier(n_estimators=i)
    # svc = SVC(C=i, kernel='rbf', degree=3, gamma='scale', random_state=1)
    scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
    bp_scores.append(scores.mean())

print(bp_scores)  # [xx, xxx, .... ]
print(max(bp_scores))
# 保存一下
pd.DataFrame(np.array(bp_scores)).to_csv('./output/十折交叉验证结果_max_depth_筛选.csv')

plt.plot(hidden_range, bp_scores)
plt.xlabel("Value of h for HiddenNum")
plt.ylabel("Cross validated accuracy")

plt.show()

