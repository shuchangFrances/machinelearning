from sklearn import svm
from sklearn import datasets

clf=svm.SVC()
iris=datasets.load_iris()
X,y=iris.data,iris.target
clf.fit(X,y)

#保存数据 method1：pickle
import  pickle
"""
#保存模型
with open('save/clf.pickle','wb') as  f:
    pickle.dump(clf,f)#把训练好的clf以.pickle形式放入文件夹
#加载模型
with open('save/clf.pickle','rb') as f:
    clf2=pickle.load(f)
    print(clf2.predict(X[0:1]))
"""
#保存数据 method2：joblib
from sklearn.externals import joblib
#save
joblib.dump(clf,'save/clf.pkl')
#restore
clf3=joblib.load('save/clf.pkl')
print(clf3.predict(X[0:1]))
