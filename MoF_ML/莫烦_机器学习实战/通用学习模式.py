import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
# sklearn.cross_validation在1.9版本以后就被弃用了，1.9版本的以后的小伙伴可以用sklearn.model_selection就行了
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


# 花的分类
# 类别：3种————标签为：0 1 2
# 每类样本： 50个
# 每类样本的特征： 4个

# 莫凡：
# # 导入数据
# iris = datasets.load_iris()
# iris_X = iris.data      # 特征向量
# iris_y = iris.target    # 种类标签
# # 查看数据
# # print(iris_X[:2, :])    # print 两个sample的属性
# # print(iris_y)   # 分类有：0 1 2这三种类型的花
#
# # 把train_data和test_data分开：
# X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
# # 把训练集和测试集分成7：3
# # print(y_train)      # 可以看出标签已经被打乱了
#
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
#
# print(knn.predict(X_test))
# print(y_test)

# Evan：  # https://zhuanlan.zhihu.com/p/33420189
from sklearn.datasets import load_iris

dataSet = load_iris(as_frame=True)
data = dataSet['data']  # 数据
label = dataSet['target']  # 数据对应的标签
feature = dataSet['feature_names']  # 特征的名称
target = dataSet['target_names']  # 标签的名称
frame = dataSet['frame']

# 查看数据集的信息
# print(target)
# print(feature)
# print(label)
# print(data)
# print(frame)

# df = pd.DataFrame(np.column_stack((data, label)), columns=np.append(feature, 'label'))
# df.head()  # 查看前五行数据; head( )函数默认只能读取前五行数据
# print(df.head(8))
#
# # 检查缺失值比例
# # df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df))
# print(df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df)))
#
# print(df['label'].value_counts())  # 检查数据类别的比例

print(StandardScaler().fit_transform(data))
# z-score 标准化将样本的特征值转换到同一量纲下，使得不同特征之间有可比性。以上就是使用了z-score标准化

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

ss = ShuffleSplit(n_splits=1, test_size=0.2)  # 按比例拆分数据，80%用作训练     ;随机打乱后划分数据集
# print(list(ss.split(data, label)))
for tr, te in ss.split(data, label):
    xr = data[tr]
    xe = data[te]
    yr = label[tr]
    ye = label[te]
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    clf.fit(xr, yr)
    predict = clf.predict(xe)
    print(classification_report(ye, predict))

# 这里我们的逻辑回归使用OVR多分类方法，
# OvR把多元逻辑回归，看做二元逻辑回归。具体做法是，每次选择一类为正例，其余类别为负例，然后做二元逻辑回归，得到第该类的分类模型。
# 最后得出多个二元回归模型。按照各个类别的得分得出分类结果。

# # model_seletion里面还提供自动调参的函数，以格搜索（GridSearchCV）为例
# clf = LogisticRegression()
# gs = GridSearchCV(clf, parameters)
# gs.fit(data, label)
# print(gs.best_params_)
