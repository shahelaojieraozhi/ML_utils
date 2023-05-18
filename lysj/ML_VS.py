import pandas as pd
import matplotlib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像负号"-"显示方框的问题

dataset = pd.read_excel('E:\\File_cache\\feiq\\Recv Files\\data_9_120.xlsx')

X = dataset.iloc[:, :9].values
y = dataset.iloc[:, -2].values
print(X.shape)
print(y.shape)

""" Random Forest Classifier """
lis = []
for i in range(10):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(X_train, y_train)
    pred_y = rfc.predict(X_test)
    #     pred_y, y_test

    # 计算准确率——自己写的
    L = pred_y.shape[0]
    flag = 0
    for i in range(L):
        if pred_y[i] == y_test[i]:
            flag += 1
    print("RF准确率为：", flag / L)
    lis.append(flag / L)
sum = 0
for i in lis:
    sum = sum + i
print(sum / len(lis))

""" DecisionTreeClassifier """
lis = []
for i in range(10):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    pred_y = clf.predict(X_test)

    # 计算准确率——自己写的
    L = pred_y.shape[0]
    flag = 0
    for i in range(L):
        if pred_y[i] == y_test[i]:
            flag += 1
    print("DT准确率为：", flag / L)
    lis.append(flag / L)
sum = 0
for i in lis:
    sum = sum + i
print('测试集平均准确率：', sum / len(lis))

""" Support vector machine """
lis = []
for i in range(10):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    svc = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', random_state=1)
    svc.fit(X_train, y_train)
    pred_y = svc.predict(X_test)

    # 计算准确率——自己写的
    L = pred_y.shape[0]
    flag = 0
    for i in range(L):
        if pred_y[i] == y_test[i]:
            flag += 1
    print("SVM准确率为：", flag / L)
    lis.append(flag / L)
sum = 0
for i in lis:
    sum = sum + i
print('测试集平均准确率：', sum / len(lis))
