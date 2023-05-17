import matplotlib.pyplot as plt
import numpy as np

# x = np.linspace(-1, 1, 50)
# # y = 2*x+1
# y = x ** 2
# plt.plot(x, y)
# plt.show()
# # home 是返回
# # 放大镜可以框选

# # 2.2 figure图像
# x = np.linspace(-3, 3, 50)
# y1 = 2 * x + 1
# y2 = x ** 2
# plt.figure()
# plt.plot(x, y1)
#
# # plt.figure()
# plt.figure(num=3)     # 改图像的序号，本来默认是figure2，改成了figure3
# plt.plot(x, y2)
# plt.show()

# # 一张图两个函数线并改变线的属性
# x = np.linspace(-3, 3, 50)
# y1 = 2 * x + 1
# y2 = x ** 2
# plt.figure(num=3)
# plt.plot(x, y1)
#
# # plt.figure()
# fig = plt.figure(figsize=(8, 4))  # 改图像的序号并且改显示尺寸(w,h)
# plt.plot(x, y2)
# plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
# fig.canvas.set_window_title('I am title')
#
# plt.show()

# # 2.3 设置到坐标轴1
# x = np.linspace(-3, 3, 50)
# y1 = 2 * x + 1
# y2 = x ** 2

# # plt.figure()
# plt.figure(num=3, figsize=(8, 5))  # 改图像的序号并且改显示尺寸
# plt.plot(x, y2)
# plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
# # 限制显示轴的范围
# plt.xlim((-1, 2))
# plt.ylim((-2, 3))
# plt.xlabel('I am x')
# plt.ylabel('I am y')
#
# new_ticks = np.linspace(-1, 2, 5)  # 只显示5个x坐标
# print(new_ticks)    # [-1.   -0.25  0.5   1.25  2.  ]
# plt.xticks(new_ticks)
#
# plt.yticks([-2, -1.8, -1, 1.22, 3, ],
#            ['really bad', 'bad', 'normal', 'good', 'really good'])
# # 在y的刻度上对应的值上标字
# plt.show()

# # 改字体
# x = np.linspace(-3, 3, 50)
# y1 = 2 * x + 1
# y2 = x ** 2
#
# # plt.figure()
# plt.figure(num=3, figsize=(8, 5))  # 改图像的序号并且改显示尺寸
# plt.plot(x, y2)
# plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
# plt.xlim((-1, 2))
# plt.ylim((-2, 3))
# plt.xlabel('I am x')
# plt.ylabel('I am y')
#
# new_ticks = np.linspace(-1, 2, 5)  # 只显示5个x坐标
# print(new_ticks)
# plt.xticks(new_ticks)
#
# plt.yticks([-2, -1.8, -1, 1.22, 3, ],
#            [r'$really\ bad$', r'$bad\ \alpha$', r'$normal$', r'$good$', r'$really\ good$'])
# # [r'$really\ bad$', r'$bad\ \alpha$', r'$normal$', r'$good$', r'$really\ good$'])  # 坐标上显示特殊符号
#
# plt.show()

# 改变坐标轴的位置
x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2

# plt.figure()
plt.figure(num=3, figsize=(8, 5))  # 改图像的序号并且改显示尺寸
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
plt.xlim((-1, 2))
plt.ylim((-2, 3))
plt.xlabel('I am x')
plt.ylabel('I am y')

new_ticks = np.linspace(-1, 2, 5)  # 只显示5个x坐标
print(new_ticks)
plt.xticks(new_ticks)

plt.yticks([-2, -1.8, -1, 1.22, 3, ],
           [r'$really\ bad$', r'$bad\ \alpha$', r'$normal$', r'$good$', r'$really\ good$'])

# gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')    # 右边的脊梁(轴),消失掉
ax.spines['top'].set_color('none')
# 视频里用了下面的.但是报错了,可能版本问题,不用也能直接输出
# ax.yaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

# 把x轴抬上来,抬到-1为起点
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
# B站弹幕: []里面存的是key,()里面存的是传入参数的一部分,这个参数是个元组
plt.show()
