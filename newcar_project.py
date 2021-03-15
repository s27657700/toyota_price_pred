import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
from sklearn.model_selection import cross_val_score
from joblib import dump, load
from newcar_transform import transform_toyota
df=pd.read_csv("price_age.csv",encoding='utf-8')
label=df['price']
df=df.drop(['price'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(df,label, test_size=0.2,random_state=0)
# displacement=int(input('排氣量 : '))
# car_model=input("車型 : ").upper()
# year=int(input("年分 : "))
# mileage=int(input('里程 : '))
# gas=input("燃料 : ")
# color=input('顏色 : ').lower()

x_train=transform_toyota(x_train)
x_test=transform_toyota(x_test)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
y_pred= model.predict(x_test)

# dump(model,'newcar_price.joblib')
# print(model.coef_)
# price:468000
test=np.array([[2000	,0.166666666666667	,0.333333333333333,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0
]])
test=transform_toyota(test)
test_pred=model.predict(test)


print(r2_score(y_test,y_pred))

print(int(test_pred))
