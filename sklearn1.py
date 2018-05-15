""" 如何载入datasets的数据集和自己创建数据集"""

from sklearn import  datasets
from sklearn.linear_model import  LinearRegression
import matplotlib.pyplot as plt
loaded_data=datasets.load_boston()
#统一的数据输入形式
data_X=loaded_data.data
data_y=loaded_data.target#也就是label

model=LinearRegression()
model.fit(data_X,data_y)

print(model.predict(data_X[:4,:]))
print(data_y[:4])

X,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)
plt.scatter(X,y)#以点的形式输出
plt.show()



