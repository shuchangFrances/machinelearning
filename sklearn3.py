"""交叉验证 """

from sklearn import  datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()
X=iris.data #iris属性
y=iris.target #iris分类

from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as  plt

k_range=range(1,31)
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    loss=-cross_val_score(knn,X,y,cv=10,scoring='mean_squared_error')#for regression 误差
    scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')#for classification
    #k_scores.append(scores.mean())
    k_scores.append(loss.mean())
plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross_Validated Accuracy')
plt.show()

#X_trian,X_test,y_train,y_test=train_test_split(X,y,random_state=4)

#knn=KNeighborsClassifier(n_neighbors=5)#考虑数据集周围五个数据
#scores=cross_val_score(knn,X,y,cv=5,scoring='accuracy')
#knn.fit(X_trian,y_train)
#print(knn.score(X_test,y_test))
#print(scores.mean())


