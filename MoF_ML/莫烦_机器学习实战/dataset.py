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

# 随机生成数据集
X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=100)
plt.scatter(X, y)
plt.show()
