import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
magic_data=pd.read_csv('magic04.txt')#读取数据
#数据是以逗号为分隔符的，但是这个数据没有列的字，所以先给每个列取个名字，直接使用数据说明中的描述
magic_data .columns=['a','b','c','d','e','f','g','h','i','j','class']
import matplotlib.patches as mpatches
from sklearn import datasets

x = magic_data['a']
mu =np.mean(x) #计算均值
sigma =np.std(x)#方差
print('μ:',mu)
print('σ²:',sigma**2)
num_bins = 30 #直方图柱子的数量
n, bins, patches = plt.hist(x, num_bins,normed=1, facecolor='blue', alpha=0.5)
#直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象
y = mlab.normpdf(bins, mu, sigma)#拟合一条最佳正态分布曲线y
plt.plot(bins, y, 'r--') #绘制y的曲线
plt.xlabel('shuxin1') #绘制x轴
plt.ylabel('Probability') #绘制y轴
plt.title(u'正太分布')#中文标题 u'xxx'
plt.subplots_adjust(left=0.15)#左边距
plt.show()