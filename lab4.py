import pandas as pd
from sklearn import tree
import graphviz
data=pd.read_csv('iris.txt')
data.columns=['1','2','3','4','class']
#提取特征和目标（前四列为特征，最后一列为类别）
feature=data[['1','2','3','4']]
targe=data[['class']]
#映入决策树模型
#选取最优节点的标准为信息增益，最小节点为5,每个类别权重至少0.95即不纯度不高于0.05
model=tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
model=model.fit(feature,targe)
tree = tree.export_graphviz(model, out_file=None,
                      feature_names=['1','2','3','4'],
                      filled=True, rounded=True,
                      special_characters=True)
graph = graphviz.Source(tree)
graph.render("iris")
