import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('./data/err_delete_features.csv')
data_copy = data.copy()
# data_copy.drop(data_copy.columns[0], axis=1, inplace=True)  # 丢弃第一列索引列

sindex = np.argsort(data_copy.isna().sum().values.tolist())  # 将有缺失值的列按缺失值的多少由小到大
# 进入for循环进行空值填补
for i in sindex:  # 按空值数量,从小到大进行排序来遍历
    if data_copy.iloc[:, i].isna().sum() == 0:  # 将没有空值的行过滤掉
        continue  # 直接跳过当前的for循环
    df = data_copy  # 复制df数据
    fillc = df.iloc[:, i]  # 将第i列的取出，之后作为y变量
    df = df.iloc[:, df.columns != df.columns[i]]  # 除了有这列以外的数据，之后作为X
    df_0 = SimpleImputer(missing_values=np.nan,  # 将df的数据全部用0填充
                         strategy="constant",
                         fill_value=0).fit_transform(df)
    Ytrain = fillc[fillc.notnull()]  # 在fillc列中,不为NAN的作为Y_train
    Ytest = fillc[fillc.isnull()]  # 在fillc列中,为NAN的作为Y_test
    Xtrain = df_0[Ytrain.index, :]  # 在df_0中(已经填充了0),中那些fillc列不为NAN的行作为Xtrain
    Xtest = df_0[Ytest.index, :]  # 在df_0中(已经填充了0),中那些fillc等于NAN的行作为X_test

    """
    Ytrain  shape: (526,)
    Ytest   shape: (1,)     value为(267, NAN)
    Xtrain  shape: (526, 55)  
    Xtest   shape: (1, 55)
    
    data 因为带了标签, 所以shape为(527, 56)
    
    这样的话相当于: 非缺失值的全部行作为训练集(526, 55), 标签是缺失行除了缺失的那个的全部值(526)
    用非缺失值去预测缺失值
    
    测试的时候用 55 个未缺失值预测那个缺失值
    """
    rfc = RandomForestRegressor()
    rfc.fit(Xtrain, Ytrain)
    Ypredict = rfc.predict(Xtest)  # Ytest为了定Xtest,以最后预测出Ypredict

    data_copy.loc[data_copy.iloc[:, i].isnull(), data_copy.columns[i]] = Ypredict
    # 将data_copy中data_copy在第i列为空值的行,第i列,改成Ypredict


pd.DataFrame(data_copy).to_csv('./data/SVM_supple_features_new.csv', header=True, index=False)


