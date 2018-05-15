"""sklearn model常用属性和功能"""

from sklearn import datasets
from sklearn.linear_model import  LinearRegression

loaded_data=datasets.load_boston()
data_X=loaded_data.data
data_y=loaded_data.target

model=LinearRegression()
model.fit(data_X,data_y)

#print(model.coef_)#如果y=0.1x+0.3，则model.coef_输出是0.1
#print(model.intercept_)#输出0.3
#print(model.get_params())#输出model的几个参数
print(model.score(data_X,data_y))#用来打分 用R^2 coefficient determination

