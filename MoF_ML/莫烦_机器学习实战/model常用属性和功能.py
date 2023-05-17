from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

print(model.predict(data_X[:4, :]))
print(data_y[:4])

# [30.00384338 25.02556238 30.56759672 28.60703649]
# [24.  21.6 34.7 33.4]

print(model.coef_)      # 如何回归模型是y=0.1x + 0.3   那么这个输出的是0.1(权重矩阵)
print(model.intercept_)     # 这个输出的是0.3
# (权重矩阵)：
# [-1.08011358e-01  4.64204584e-02  2.05586264e-02  2.68673382e+00
#  -1.77666112e+01  3.80986521e+00  6.92224640e-04 -1.47556685e+00
#   3.06049479e-01 -1.23345939e-02 -9.52747232e-01  9.31168327e-03 -5.24758378e-01]
# 偏移量：
# 36.45948838509001

print(model.get_params())       # 找之前定义的参数，没定义的返回默认值
# {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'normalize': 'deprecated', 'positive': False}

print(model.score(data_X, data_y))      # LinearRegression 线性回归模型里面打分的依据用的是
# 0.7406426641094094  ——————拟合优度
# 拟合优度（Goodness of Fit）是指回归直线对观测值的拟合程度。度量拟合优度的统计量是可决系数（亦称确定系数）R²。R²最大值为1。R²的值越接近1，
# 说明回归直线对观测值的拟合程度越好；反之，R²的值越小，说明回归直线对观测值的拟合程度越差。
