import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#

# # Series
# data = pd.Series(np.random.randn(1000), index=np.arange(1000))
# # cunsum是把生成的一千个数字累加
# data = data.cumsum()
# data.plot()
# plt.show()

# # DataFrame
data = pd.DataFrame(np.random.randn(1000, 4),
                    index=np.arange(1000),
                    columns=list("ABCD"))
print(data)
data = data.cumsum()
print(data)
data.plot()
plt.show()

# plot methods:
# 'bar','hist','box','kde','area','scatter','hexbin','pie'
ax = data.plot.scatter(x='A', y='B', color='DarkBlue', label='Class 1')
data.plot.scatter(x='A', y='C', color='DarkGreen', label='Class 2', ax=ax)
plt.show()

