from sklearn.learning_curve import learning_curve #可视化整个过程
from sklearn.datasets import load_digits #数字
from sklearn.svm import SVC #支持向量机
import  matplotlib.pyplot as plt
import numpy as np

digits=load_digits()
X=digits.data
y=digits.target

train_sizes,train_loss,test_loss=learning_curve(
    SVC(gamma=0.1),X,y,cv=10,scoring='mean_squared_error',
    train_sizes=[0.1,0.25,0.5,0.75,1]#10%记录一下 25%记录 打上点
)
train_loss_mean=-np.mean(train_loss,axis=1)#平均误差
test_loss_mean=-np.mean(test_loss,axis=1)

plt.plot(train_sizes,train_loss_mean,'o-',color="r",
         label="Training")
plt.plot(train_sizes,test_loss_mean,'o-',color="g",
         label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()