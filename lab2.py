import numpy as np
import csv
import pandas as pd

data=pd.read_csv('iris.txt')
data=np.array(data)
data=np.mat(data[:,0:4])#提取前四列属性
length=len(data)
#通过核函数得到核矩阵
k=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k[i,j]=(np.dot(data[i],data[j].T))**2
        k[j,i]=k[i,j]


len_k=len(k)
I=np.eye(len_k)
one=np.ones((len_k,len_k))
A=I-1.0/len_k*one
centered_k=np.dot(np.dot(A,k),A)#居中核矩阵

N=np.zeros((len_k,len_k))
for i in range(0,len_k):
    N[i,i]=centered_k[i,i]**(-0.5)
normalized_k=np.dot(np.dot(N,centered_k),N)#归一化矩阵（规范核）
name=range(length)
test=pd.DataFrame(columns=name,data=normalized_k)
test.to_csv('iris_1.csv')#标准化核矩阵
#顺序相反操作
#核函数展开得到矩阵
H=np.mat(np.zeros((length,10)))
for i in range(0,length):
    for j in range(0,4):
        H[i,j]=data[i,j]**2
    for m in range(0,3):
        for n in range(m+1,4):
            j=j+1
            H[i,j]=2**0.5*data[i,m]*data[i,n]
#矩阵中心化
nameH=range(10)
rows=H.shape[0]
cols=H.shape[1]
centered_H=np.mat(np.zeros((rows,cols)))
for i in range(0,cols):
    centered_H[:,i]=H[:,i]-np.mean(H[:,i])
#中心化的矩阵在归一化
normalized_H=np.mat(np.zeros((rows,cols)))
for i in range(0,len(H)):
    normalized_H[i]=centered_H[i]/np.linalg.norm(centered_H[i])
#最后得到的矩阵转化为核矩阵
H1=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        H1[i,j]=(np.dot(normalized_H[i],normalized_H[j].T))
        H1[j,i]=H1[i,j]
test=pd.DataFrame(columns=name,data=H1)
test.to_csv('iris_2.csv')