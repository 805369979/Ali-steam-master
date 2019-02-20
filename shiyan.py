
#  SimHei中文黑体 Kaiti中文楷体 LiSu中文隶书 FangSong中文仿宋
#  YouYuan中文幼圆 STSong华文宋体

# matplotlib.rcParams['font.family'] = 'SimHei'

# 绘制一个坐标图
# plt.plot([3, 1, 4, 5, 2])
# plt.xlabel("横轴(值)", fontproperties='SimHei', fontsize=20)
# plt.ylabel("纵轴(值)", fontproperties='SimHei', fontsize=20)
# plt.savefig('test', dip=600)
# plt.show()

# 绘制一个箱型图
# plt.boxplot(x, notch=None, sym=None, vert=None, whis=None, positions=None,
#             widths=None, patch_artist=None, meanline=None, showmeans=None,
#             showcaps=None, showbox=None, showfliers=None, boxprops=None,
#             labels=None, flierprops=None, medianprops=None, meanprops=None,
#             capprops=None, whiskerprops=None)
# x：指定要绘制箱线图的数据；
# notch：是否是凹口的形式展现箱线图，默认非凹口；
# sym：指定异常点的形状，默认为+号显示；
# vert：是否需要将箱线图垂直摆放，默认垂直摆放；
# whis：指定上下须与上下四分位的距离，默认为1.5倍的四分位差；
# positions：指定箱线图的位置，默认为[0,1,2…]；
# widths：指定箱线图的宽度，默认为0.5；
# patch_artist：是否填充箱体的颜色；
# meanline：是否用线的形式表示均值，默认用点来表示；
# showmeans：是否显示均值，默认不显示；
# showcaps：是否显示箱线图顶端和末端的两条线，默认显示；
# showbox：是否显示箱线图的箱体，默认显示；
# showfliers：是否显示异常值，默认显示；
# boxprops：设置箱体的属性，如边框色，填充色等；
# labels：为箱线图添加标签，类似于图例的作用；
# filerprops：设置异常值的属性，如异常点的形状、大小、填充色等；
# medianprops：设置中位数的属性，如线的类型、粗细等；
# meanprops：设置均值的属性，如点的大小、颜色等；
# capprops：设置箱线图顶端和末端线条的属性，如颜色、粗细等；
# whiskerprops：设置须的属性，如颜色、粗细、线的类型等；


# 绘制一个条形图
# 垂直
# plt.bar(left, height, alpha=1, width=0, 8, color=, edgecolor=, label=, lw=3)
# left：x轴的位置序列，一般采用arange函数产生一个序列；
# height：y轴的数值序列，也就是柱形图的高度，一般就是我们需要展示的数据；
# alpha：透明度
# width：为柱形图的宽度，一般这是为0.8即可；
# color或facecolor：柱形图填充的颜色；
# edgecolor：图形边缘颜色
# label：解释每个图像代表的含义
# linewidth or linewidths or lw：边缘or线的宽度
# 水平
# plt.barh([1, 2, 3, 4], [2, 2, 2, 2], 0.8, None)
# plt.show()

# 绘制一个极坐标图


# # 原始数据
# X = [1, 2, 3, 4, 5, 6]
# Y = [2.6, 3.4, 4.7, 5.5, 6.47, 7.8]
#
# # 用一次多项式拟合，相当于线性拟合
# z1 = np.polyfit(X, Y, 1)
# p1 = np.poly1d(z1)
# print(z1)  # [ 1.          1.49333333]
# print(p1)  # 1 x + 1.493

# file_path = r'C:\Users\zugle\Desktop\Python_code\steam\zhengqi_train.txt'
# # a = pd.read_csv(file_path, sep=' ')
# a = pd.read_csv(file_path, delim_whitespace=True)
# b = pd.Series(np.array([1, 2]), index=["a", "b"])
# c = pd.Series(np.array([3, 4]), index=["b", "d"])
# print(b + c)
# # print(a.index)

