import  numpy as np
from sklearn import  datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()
iris_X=iris.data #iris属性
iris_y=iris.target #iris分类

#print(iris_X[:2,:])
#print(iris_y) #三个类别

X_trian,X_test,y_train,y_test=train_test_split(iris_X,iris_y,test_size=0.3)
#将数据集分类 30%测试集 打乱了数据集
#print(y_train)

knn=KNeighborsClassifier()
knn.fit(X_trian,y_train)
print(knn.predict(X_test))#用我的模型来预测测试集结果
print(y_test) #真实结果


