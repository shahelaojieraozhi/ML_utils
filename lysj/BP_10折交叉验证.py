from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import joblib
from itertools import cycle
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像负号"-"显示方框的问题

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
# hidden_range = range(1, 31)
# bp_scores = []
# for h in hidden_range:
#     mlp = MLPClassifier(hidden_layer_sizes=(h), max_iter=1000)
#     scores = cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')
#     bp_scores.append(scores.mean())
#
# print(bp_scores)  # [xx, xxx, .... ]
# print(max(bp_scores))
# # 保存一下
# pd.DataFrame(np.array(bp_scores)).to_csv('./output/十折交叉验证结果_BP_筛选hidden.csv')
#
# plt.plot(hidden_range, bp_scores)
# plt.xlabel("Value of h for HiddenNum")
# plt.ylabel("Cross validated accuracy")
#
# plt.show()

'画学习曲线'
train_sizes, train_loss, test_loss = learning_curve(
    MLPClassifier(hidden_layer_sizes=(6), max_iter=1000), X_train, y_train, cv=10, scoring='neg_mean_squared_error',
    train_sizes=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                 0.95, 1])
# train_sizes=[0.1, 0.25, 0.5, 0.75, 1] train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)
lc = pd.concat([pd.DataFrame(train_loss_mean), pd.DataFrame(test_loss_mean)], axis=1)
lc.to_csv('./output/lc_BP_6.csv')

plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
         label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()

bp = MLPClassifier(hidden_layer_sizes=(6), max_iter=1000)
bp.fit(X_train, y_train)
y_predict_bp = bp.predict(X_test)
print("准确率：", accuracy_score(y_test, y_predict_bp))  # 其实是调参后

'画ROC曲线'
# Compute ROC curve and ROC area for each class


bp.fit(X_train, y_train)
y_score = bp.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()  # 定义三个字典


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


Y_onehot = to_categorical(y_test)

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(Y_onehot[:, i], y_score[:, i])  # y_test[:,i] 第i列
    roc_auc[i] = auc(fpr[i], tpr[i])

    """
    roc_curve()
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=False)

    主要参数：
    y_true：真实的样本标签，默认为{0，1}或者{-1，1}。如果要设置为其它值，则 pos_label 参数要设置为特定值。例如要令样本标签为{1，2}，其中2表示正样本，则pos_label=2。
    y_score：对每个样本的预测结果。
    pos_label：正样本的标签。

    返回值：
    fpr：False positive rate。
    tpr：True positive rate。
    thresholds
    """

# 求得所有的fpr
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))  # concatenate是内置拼接函数 unique返回一个无元素重复的数组或列表
# 在该点插入所有的roc曲线
mean_tpr = np.zeros_like(all_fpr)  # 生成一个跟all_fpr相同类型的全为0的数组
for i in range(3):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])  # interp一维线性插值，all_fpr表示待插入数据横坐标，fpr[i]、tpr[i]原始数据横纵坐标
    # 求平均值并求auc
mean_tpr /= 3
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])  # macro是参数，求多分类ROC不同方法参数不同

# 绘制所有的roc曲线
lw = 2
plt.figure()
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)  # 绘制平均roc并附加说明

colors = cycle(['aqua', 'darkorange', 'darkkhaki'])
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))  # 绘制三分类roc并附加说明


a = fpr["macro"]
b = tpr["macro"]

ROC_plt = pd.concat([pd.DataFrame(a), pd.DataFrame(b)], axis=1)
ROC_plt.to_csv('./output/ROC_plt.csv', float_format='%.5f', header=False, index=False)


plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')  # x轴说明
plt.ylabel('True Positive Rate')  # y轴说明
plt.title('Some extension of Receiver operating characteristic to multi-class')  # 标题
plt.legend(loc="lower right")
plt.show()

print(fpr)
print(tpr)

cnf_matrix = confusion_matrix(y_test, y_predict_bp)
# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(20, 8), dpi=80)
ax.matshow(cnf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for a in range(cnf_matrix.shape[0]):
    for b in range(cnf_matrix.shape[1]):
        ax.text(x=b, y=a, s=cnf_matrix[a, b], va="center", ha="center", fontsize=15)
plt.title('Confusion matrix -- RandomForestClassifier')
plt.ylabel('Actual value', fontsize=12)
plt.xlabel('Predictive value', fontsize=12)
plt.tick_params(labelsize=12)
# #plt.savefig("./rf_3_matirx.png")
plt.show()

pd.DataFrame(cnf_matrix).to_csv('./output/cnf_matrix.csv')

print(classification_report(y_predict_bp, y_test))

