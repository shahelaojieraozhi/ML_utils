'参考：https://blog.csdn.net/tz_zs/article/details/81069499'
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

# 创建一个子图
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

plt.show()

# 创建两个子图，并且共享y轴
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)

plt.show()

# 创建4个子图，
fig1, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))  # polar:极地
axes[0, 0].plot(x, y)
axes[1, 1].scatter(x, y)
plt.show()

