import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

""" 我们使用了LightGBM的默认参数 """

data = pd.read_csv("data/SVM_supple_features.csv")

# 将标签列提取出来，作为y
y = data['Discomfortrating'].values

# 剔除标签列，作为X
X = data.drop('Discomfortrating', axis=1).values

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# 训练模型
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
model = lgb.train(params, lgb_train, valid_sets=lgb_test, num_boost_round=1000, early_stopping_rounds=50)

pre = model.predict(X_test)
print('score : ', np.mean((pre > 0.5) == y_test))

# 分析特征重要性
# importance = pd.DataFrame(model.feature_importance(importance_type='gain'), index=data.columns[:-1],
#                           columns=['importance'])
importance = pd.DataFrame(model.feature_importance(), index=data.columns[:-1],
                          columns=['importance'])
importance.sort_values(by='importance', ascending=False, inplace=True)
print(importance)

# 剔除重要性较低的特征
low_importance_features = importance[importance['importance'] < 15].index.tolist()
data.drop(low_importance_features, axis=1, inplace=True)

data.to_csv('./data/drop_low_importance_features.csv', index=False)

# 重新划分X和y
y = data['Discomfortrating'].values
X = data.drop('Discomfortrating', axis=1).values

# 重新训练模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
model = lgb.train(params, lgb_train, valid_sets=lgb_test, num_boost_round=1000, early_stopping_rounds=50)

pre = model.predict(X_test)
print('score : ', np.mean((pre > 0.5) == y_test))

# 分析特征重要性
importance = pd.DataFrame(model.feature_importance(), index=data.columns[:-1],
                          columns=['importance'])
importance.sort_values(by='importance', ascending=False, inplace=True)
print(importance)
