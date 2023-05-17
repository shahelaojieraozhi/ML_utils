import matplotlib.pyplot as plt
import numpy as np

# # 改变坐标轴的位置
# x = np.linspace(-3, 3, 50)
# y1 = 2 * x + 1
# y2 = x ** 2
#
# # plt.figure()
# plt.figure(num=3, figsize=(8, 5))  # 改图像的序号并且改显示尺寸
#
# plt.xlim((-1, 2))   # x有(-3,3)但是坐标轴上只显示(-1,2)
# plt.ylim((-2, 3))
# plt.xlabel('I am x')    # 设置x的label的名字
# plt.ylabel('I am y')
#
# new_ticks = np.linspace(-1, 2, 5)  # 只显示5个x坐标
# print(new_ticks)
# plt.xticks(new_ticks)
#
# plt.yticks([-2, -1.8, -1, 1.22, 3, ],
#            [r'$really\ bad$', r'$bad\ \alpha$', r'$normal$', r'$good$', r'$really\ good$'])
#
# plt.plot(x, y2, label='up')
# plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='down')
# plt.legend()
# # plt.legend(handles=[], labels=, loc='best')
#
# plt.show()

# # 改legend的细节
# x = np.linspace(-3, 3, 50)
# y1 = 2 * x + 1
# y2 = x ** 2
#
# plt.figure(num=3, figsize=(8, 5))  # 改图像的序号并且改显示尺寸
#
# plt.xlim((-1, 2))   # x有(-3,3)但是坐标轴上只显示(-1,2)
# plt.ylim((-2, 3))
# plt.xlabel('I am x')    # 设置x的label的名字
# plt.ylabel('I am y')
#
# new_ticks = np.linspace(-1, 2, 5)  # 只显示5个x坐标
# print(new_ticks)
# plt.xticks(new_ticks)
#
# plt.yticks([-2, -1.8, -1, 1.22, 3, ],
#            [r'$really\ bad$', r'$bad\ \alpha$', r'$normal$', r'$good$', r'$really\ good$'])
#
# l1, = plt.plot(x, y2, label='up')
# l2, = plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='down')
#
# plt.legend(handles=[l1, l2, ], labels=['aaa', 'bbb'], loc='best')
# # 第一个参数handles:在legend里显示需要的对象
# # 第二个参数label:显示legend里label的名字(如:上面label的名字是'up'这里可以改成"aaa")
# # 第三个参数loc:显示的位置,'best'是程序自动找到最好的位置放legend
#
# plt.show()


# # Anotation
# x = np.linspace(-3, 3, 50)
# y = 2 * x + 1
# plt.figure(num=1, figsize=(8, 5))  # 改图像的序号并且改显示尺寸
# plt.plot(x, y)
# # plt.scatter(x, y)   # 画散点图的
#
# ax = plt.gca()
# ax.spines['right'].set_color('none')    # 右边的脊梁(轴),消失掉
# ax.spines['top'].set_color('none')
#
# # 把x轴抬上来,抬到-1为起点
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['left'].set_position(('data', 0))
# # B站弹幕: []里面存的是key,()里面存的是传入参数的一部分,这个参数是个元组
#
# # 画(1,0)到(1,y0)的黑色虚线
# x0 = 1
# y0 = 2 * x0 + 1
# plt.scatter(x0, y0, s=50, color='r')    # 那个大红点
# plt.plot([x0, x0], [y0, 0], 'k--', lw=2.5)
#
# # 画箭头了:
# plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data',
#              xytext=(+30, -30), textcoords='offset points', fontsize=16,
#              arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.2'))
#
# # 显示红字
# plt.text(-3.7, 3, r'$This\ is\ the\ some\ text.\ \mu \sigma_i\ \alpha_t$',
#          fontdict={'size': 16, 'color': 'r'})
# plt.show()


# 透明度
x = np.linspace(-3, 3, 50)
y = 0.1 * x

plt.figure()
plt.plot(x, y, linewidth=10)
plt.ylim(-2, 2)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)  # 坐标轴的字体大小
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.2))
plt.show()
