from matplotlib.pyplot import MultipleLocator
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.mlab as mlab
from pylab import *
import matplotlib.patches as mpatches
from sklearn import datasets
mpl.rcParams['font.sans-serif'] = ['SimHei']
magic_data=pd.read_csv('magic04.txt')#读取数据
#数据是以逗号为分隔符的，但是这个数据没有列的字，所以先给每个列取个名字，直接使用数据说明中的描述
magic_data .columns=['a','b','c','d','e','f','g','h','i','j','class']
colNum=magic_data.shape
print(colNum)#列数
magic_data.drop(["class"],axis=1,inplace=True)
# a=magic_data['a'].sum()/19020
# b=magic_data['b'].sum()/19020
# c=magic_data['c'].sum()/19020
# d=magic_data['d'].sum()/19020
# e=magic_data['e'].sum()/19020
# f=magic_data['f'].sum()/19020
# g=magic_data['g'].sum()/19020
# h=magic_data['h'].sum()/19020
# i=magic_data['i'].sum()/19020
# j=magic_data['j'].sum()/19020
# print('均值向量:',a,b,c,d,e,f,g,h,i,j)
#
# aa=np.array([magic_data['a'],magic_data['b'],magic_data['c'],magic_data['d'],magic_data['e'],magic_data['f'],magic_data['g'],magic_data['h'],magic_data['i'],magic_data['j']])
# bb=aa.T
# cc=np.dot(aa,bb)/19019

# print('矩阵内积:',cc)
print(magic_data.iloc[0])


a1={}
b1={}
c1={}
d1={}
for i in range(19018):
    a1=magic_data.iloc[i]
    b1=a1.T
    c1=np.outer(a1,b1)
    d1=+c1
print('矩阵外积:',d1/19019)


# # 取出iris数据中的第0列，即表示花萼长度
# x = magic_data['a']  # x轴
# y = magic_data['b']  # y轴   花萼宽度
#
#
#
# # 计算散点图x轴最小值，最大值
# x_min, x_max = x.min() - .5, x.max() + .5
# # 计算散点图y轴最小值与最大值
# y_min, y_max = y.min() - .5, y.max() + .5
# # 以下绘制散点图
# plt.figure()
#
#
#
# plt.title(u'花萼长宽散点图及均值')
# plt.scatter(x, y,c='r')
# plt.xlabel(u' attributes1')
# plt.ylabel(u' attributes2')
#
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(x,fontsize=9)
# plt.yticks(y)
# plt.plot()
#
# x_major_locator=MultipleLocator(20)
# #把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator(20)
# #把y轴的刻度间隔设置为10，并存在变量里
# ax=plt.gca()
# #ax为两条坐标轴的实例
# ax.xaxis.set_major_locator(x_major_locator)
# #把x轴的主刻度设置为1的倍数
# ax.yaxis.set_major_locator(y_major_locator)
# #把y轴的主刻度设置为10的倍数
# plt.xlim(0,130)
# #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
# plt.ylim(0,110)
# #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
#
# plt.show()
cor1=magic_data['a'].corr(magic_data['b'])
print('相关系数:',cor1)

sigma1=np.std(magic_data['a'])
sigma2=np.std(magic_data['b'])
sigma3=np.std(magic_data['c'])
sigma4=np.std(magic_data['d'])
sigma5=np.std(magic_data['e'])
sigma6=np.std(magic_data['f'])
sigma7=np.std(magic_data['g'])
sigma8=np.std(magic_data['h'])
sigma9=np.std(magic_data['i'])
sigma10=np.std(magic_data['j'])
cov1=np.cov(magic_data['a'])
cov2=np.cov(magic_data['b'])
cov3=np.cov(magic_data['c'])
cov4=np.cov(magic_data['d'])
cov5=np.cov(magic_data['e'])
cov6=np.cov(magic_data['f'])
cov7=np.cov(magic_data['g'])
cov8=np.cov(magic_data['h'])
cov9=np.cov(magic_data['i'])
cov10=np.cov(magic_data['j'])
print('属性1方差：',sigma1,'属性1协方差：',cov1)
print('属性2方差：',sigma2,'属性2协方差：',cov2)
print('属性3方差：',sigma3,'属性3协方差：',cov3)
print('属性4方差：',sigma4,'属性4协方差：',cov4)
print('属性5方差：',sigma5,'属性5协方差：',cov5)
print('属性6方差：',sigma6,'属性6协方差：',cov6)
print('属性7方差：',sigma7,'属性7协方差：',cov7)
print('属性8方差：',sigma8,'属性8协方差：',cov8)
print('属性9方差：',sigma9,'属性9协方差：',cov9)
print('属性10方差：',sigma10,'属性10协方差：',cov10)