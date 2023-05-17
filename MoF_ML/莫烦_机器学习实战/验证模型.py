from sklearn import datasets
from sklearn.linear_model import LinearRegression


loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)
# fit(x,y) :传入数据以及标签即可训练模型，训练的时间和参数设置，数据集大小以及数据本身的特点有关

print(model.predict(data_X[:4, :]))
# predict(x)用于对数据的预测，它接受输入，并输出预测标签，输出的格式为numpy数组。我们通常使用这个方法返回测试的结果，再将这个结果用于评估模型。

print(model.coef_)
print(model.intercept_)
print(model.get_params())
print(model.score(data_X, data_y))  # R^2 coefficient of determination
# score(x,y)用于对模型的正确率进行评分(范围0-1)。但由于对在不同的问题下，评判模型优劣的的标准不限于简单的正确率，
# 可能还包括召回率或者是查准率等其他的指标，特别是对于类别失衡的样本，准确率并不能很好的评估模型的优劣，因此在对模型进行评估时，不要轻易的被score的得分蒙蔽。

# 莫凡理解过拟合：期末考试专门复习平常做的习题，都会做了。考试都是新题，完了
