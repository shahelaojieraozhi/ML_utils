import matplotlib.pyplot as plt
import numpy as np

# 3.7 散点图
n = 1024
X = np.random.normal(0, 1, n)   # 钟形分布均值为0标准差
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X)  # for color value 这个函数不需要深入
plt.scatter(X, Y, s=120, c=T, alpha=0.4)
# 参数说明：
# s：是散点的大小； c就是颜色； alpha是透明度(值越小越透明)；

# 显示两个点
# plt.scatter(np.arange(5),np.arange(5))

plt.xlim((-1.5, 1.5))
plt.xlim((-1.5, 1.5))

# # 隐藏x，y轴的标注
# plt.xticks(())
# plt.yticks(())

plt.show()

# 条形图
n = 12
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

# print(np.random.uniform(0.5, 1.0, n))
# 生成n个0.5到1.0的随机数
# [0.94379506 0.54020561 0.69420618 0.77696505 0.92495455 0.87030901
# 0.99222507 0.86920711 0.61241698 0.91201769 0.73266228 0.9310038 ]

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
for x, y in zip(X, Y1):
    # ha: horizontal alignment  对齐方式
    # va: vertical alignment    纵向对齐方式
    plt.text(x, y + 0.02, '%.2f' % y, ha='center', va='bottom')   # %.2f 保留两个小数点

for x, y in zip(X, Y2):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x, -y - 0.02, '-%.2f' % y, ha='center', va='top')
plt.xlim(-.5, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())

plt.show()


# # contous 等高线图
# def f(x, y):
#     # the height function
#     return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
#
#
# n = 256
# x = np.linspace(-3, 3, n)
# y = np.linspace(-3, 3, n)
# X, Y = np.meshgrid(x, y)
#
# # use plt.contourf to filling contours
# # X, Y and value for (X,Y) point
# plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
# # 8：分成10个部分————————如果这里是0则分成2块区域
# # alpha=.75 透明度为0.5
# # cmap=plt.cm.cool   z对应的颜色，这里是冷色调
#
# # use plt.contour to add contour lines
# C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
# # adding label
# plt.clabel(C, inline=True, fontsize=10)     # 把登高线上的值标在线上
# # inline=True 表示显示值的周围有空的地方，好看点   =False的话表示线直接穿过了数字
#
# plt.xticks(())
# plt.yticks(())
# plt.show()


# # image 图片
# a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
#               0.365348418405, 0.439599930621, 0.525083754405,
#               0.423733120134, 0.525083754405, 0.651536351379]).reshape(3, 3)    # image data
#
# """
# for the value of "interpolation", check this:
# http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
# for the value of "origin"= ['upper', 'lower'], check this:
# http://matplotlib.org/examples/pylab_examples/image_origin.html
# """
# plt.imshow(a, interpolation='nearest', cmap='bone', origin='upper')
# # interpolation='nearest'：网站里可以看到很多类型，这种是最分明的那种
# # origin='upper'——————按照矩阵的顺序显示image    origin='lower' ——————相反的方向排列
#
# # 这是右边的标注
# plt.colorbar(shrink=.92)    # shrink=.92————把标注压缩
#
# plt.xticks(())
# plt.yticks(())
# plt.show()

# # 3d 数据
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = Axes3D(fig)
# # X, Y value
# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X ** 2 + Y ** 2)
# # height value
# Z = np.sin(R)
#
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# # cmap=plt.get_cmap('rainbow')————————颜色图
# # rstride=2, cstride=1——————行列跨度，只能为整数
# """
# ============= ================================================
#         Argument      Description
#         ============= ================================================
#         *X*, *Y*, *Z* Data values as 2D arrays
#         *rstride*     Array row stride (step size), defaults to 10
#         *cstride*     Array column stride (step size), defaults to 10
#         *color*       Color of the surface patches
#         *cmap*        A colormap for the surface patches.
#         *facecolors*  Face colors for the individual patches
#         *norm*        An instance of Normalize to map values to colors
#         *vmin*        Minimum value to map
#         *vmax*        Maximum value to map
#         *shade*       Whether to shade the facecolors
#         ============= ================================================
# """
#
# # I think this is different from plt12_contours
# # 画等高线
# ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
# # zdir='z'————表示等高线从哪里压下去
# # offset=-4  对于z：就是压到z=-4这个平面上去
# """
# ==========  ================================================
#         Argument    Description
#         ==========  ================================================
#         *X*, *Y*,   Data values as numpy.arrays
#         *Z*
#         *zdir*      The direction to use: x, y or z (default)
#         *offset*    If specified plot a projection of the filled contour
#                     on this position in plane normal to zdir
#         ==========  ================================================
# """
#
# ax.set_zlim(-2, 2)
#
# plt.show()

# subplot  #############

# # example 1:
# ###############################
# plt.figure(figsize=(6, 4))
# # plt.subplot(n_rows, n_cols, plot_num)
# plt.subplot(2, 2, 1)
# plt.plot([0, 1], [0, 1])
#
# plt.subplot(222)
# plt.plot([0, 1], [0, 2])
#
# plt.subplot(223)
# plt.plot([0, 1], [0, 3])
#
# plt.subplot(224)
# plt.plot([0, 1], [0, 4])
#
# plt.tight_layout()

# # example 2:
# ###############################
# plt.figure(figsize=(6, 4))
# # plt.subplot(n_rows, n_cols, plot_num)
# plt.subplot(2, 1, 1)
# # figure splits into 2 rows, 1 col, plot to the 1st sub-fig
# plt.plot([0, 1], [0, 1])
#
# plt.subplot(234)
# # figure splits into 2 rows, 3 col, plot to the 4th sub-fig
# plt.plot([0, 1], [0, 2])
#
# plt.subplot(235)
# # figure splits into 2 rows, 3 col, plot to the 5th sub-fig
# plt.plot([0, 1], [0, 3])
#
# plt.subplot(236)
# # figure splits into 2 rows, 3 col, plot to the 6th sub-fig
# plt.plot([0, 1], [0, 4])
#
# plt.tight_layout()
# plt.show()


# grid_subplot

# method 1: subplot2grid
#########################
# plt.figure()
# ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)  # stands for axes
# # (3, 3)：整个图像分成3*3的网格
# # (0, 0)： 起点是多少
# # colspan: 是列跨度，colspan=3 相当于跨了三列  ——————对于3*3的最高是3，colspan=100也是3的效果
# # rowspan=2 :是行跨度
#
# ax1.plot([1, 2], [1, 2])
# ax1.set_title('ax1_title')
# # 设置x坐标范围改成了： ax1.set_xlim()
# # 使用网格显示，后面操作都要ax1.set_....
#
# ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
# ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
# ax4 = plt.subplot2grid((3, 3), (2, 0))
# ax4.scatter([1, 2], [2, 2])
# ax4.set_xlabel('ax4_x')
# ax4.set_ylabel('ax4_y')
# ax5 = plt.subplot2grid((3, 3), (2, 1))
# plt.show()

import matplotlib.gridspec as gridspec

# method 2: gridspec
#########################
# plt.figure()
# gs = gridspec.GridSpec(3, 3)
# # use index from 0
# ax6 = plt.subplot(gs[0, :])     # 占了第一行
# ax7 = plt.subplot(gs[1, :2])    # 第二行第一列到第二列
# ax8 = plt.subplot(gs[1:, 2])
# ax9 = plt.subplot(gs[-1, 0])
# ax10 = plt.subplot(gs[-1, -2])
#
# plt.show()

# # method 3: easy to define structure
# ####################################
# f, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, sharex=True, sharey=True)
# ax11.scatter([1, 2], [1, 2])
#
# plt.tight_layout()
# plt.show()

# # 画中画
# fig = plt.figure()
# x = [1, 2, 3, 4, 5, 6, 7]
# y = [1, 3, 4, 2, 5, 8, 6]
#
# # below are all percentage
# left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# # 设置figure的大小，0.1百分比
# ax1 = fig.add_axes([left, bottom, width, height])  # main axes
#
# ax1.plot(x, y, 'r')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_title('title')
#
# # 左上角加个小图
# ax2 = fig.add_axes([0.2, 0.6, 0.25, 0.25])  # inside axes
# ax2.plot(y, x, 'b')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.set_title('title inside 1')
#
#
# # different method to add axes
# ####################################
# plt.axes([0.6, 0.2, 0.25, 0.25])
# plt.plot(y[::-1], x, 'g')       # 提醒一下：y[::-1] y里的值逆序一下
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('title inside 2')
#
# plt.show()


# #  secondary_yaxis
# x = np.arange(0, 10, 0.1)
# y1 = 0.05 * x**2
# y2 = -1 *y1
#
# fig, ax1 = plt.subplots()
#
# ax2 = ax1.twinx()    # mirror the ax1
# ax1.plot(x, y1, 'g-')
# ax2.plot(x, y2, 'b-')
#
# ax1.set_xlabel('X data')
# ax1.set_ylabel('Y1 data', color='g')
# ax2.set_ylabel('Y2 data', color='b')
#
# plt.show()


# # animation
# from matplotlib import pyplot as plt
# from matplotlib import animation
#
# fig, ax = plt.subplots()
#
# x = np.arange(0, 2 * np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))
#
# # 第i帧
#
#
# def animate(i):
#     line.set_ydata(np.sin(x + i / 10.0))  # update the data
#     return line,
#
#
# # Init only required for blitting to give a clean slate.
# def init():
#     line.set_ydata(np.sin(x))
#     return line,
#
#
# # call the animator.  blit=True means only re-draw the parts that have changed.
# # blit=True dose not work on Mac, set blit=False
# # interval= update frequency
# ani = animation.FuncAnimation(fig=fig, func=animate, frames=100, init_func=init,
#                               interval=20, blit=True)
# # frames=100 100帧
# # init_func=init 动画最开始的样子   interval=20  update 频率（毫秒）  blit=False————更新变化的点  blit=False————更新整个
#
# # save the animation as an mp4.  This requires ffmpeg or mencoder to be
# # installed.  The extra_args ensure that the x264 codec is used, so that
# # the video can be embedded in html5.  You may need to adjust this for
# # your system: for more information, see
# # http://matplotlib.sourceforge.net/api/animation_api.html
# # ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
#
# plt.show()
