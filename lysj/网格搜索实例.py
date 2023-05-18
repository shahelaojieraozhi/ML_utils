import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV

# 参考：https://www.cnblogs.com/volcao/p/9291831.html

digits = datasets.load_digits()
X = digits.data
y = digits.target

'''
三、网格搜索
其实整个交叉验证的过程在网格搜索过程中已经被使用过；
网格搜索过程，其中 CV 就是指 Cross Validation（交叉验证）；
网格搜索返回了一个 GridSearchCV 的对象，这个对象并不是最佳的算法模型，只是包含了搜索的结果，
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=666)

param_grid = [
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(2, 11)],
        'p': [i for i in range(1, 6)]
    }
]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, verbose=1, cv=3)
grid_search.fit(X_train, y_train)

# 输出：Fitting 3 folds for each of 45 candidates, totalling 135 fits
'''
“3 folds”：就是指网格搜索使用交叉验证的方式，默认将 X_train 分割成 3 份；如果将 X_train 分割成 k 份：grid_search = GridSearchCV(knn_clf, param_grid, verbose=1, cv=k)；
“45 candidates”：k 从 [2, 11) 有 9 个值，p 从 [1, 6) 有 5 个值，一共要对 45 组参数进行搜索；
“135 fits”：一共要对 45 组参数进行搜索，每次搜索要训练出 3 个模型，一共要进行 135 此拟合训练；
'''

# 查看在网格搜索过程中，交叉验证的最佳的准确率
print('查看在网格搜索过程中，交叉验证的最佳的准确率:', grid_search.best_score_)
# 输出：0.9823747680890538

# 查看搜索找到的最佳超参数
print('查看搜索找到的最佳超参数:', grid_search.best_params_)
# 输出：{'n_neighbors': 2, 'p': 2, 'weights': 'distance'}

# 获得最佳参数对应的最佳分类器（也就是最佳模型），此分类器不需要再进行 fit
best_knn_clf = grid_search.best_estimator_
best_knn_clf.score(X_test, y_test)
print('最佳参数对应的最佳分类器的分类效果:', best_knn_clf.score(X_test, y_test))
# 输出：0.980528511821975


