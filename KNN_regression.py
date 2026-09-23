import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('house_price.csv')
df=df.drop(columns=['Location','Condition','Garage'])
df=df.dropna()
df=df.drop_duplicates()
df_y=df['Price']
df_X=df.drop(columns=['Price'])
X=df_X.to_numpy()
y=df_y.to_numpy()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
k=1
def distance(a,b):
	d=0
	for i in range(6):
		d=d+(a[i]-b[i])**2
	return d
y_pred=[]
for y in X_test:
	l=[]
	for id,z in enumerate(X_train):
		d=distance(y,z)
		l.append((id,d))
	l.sort(key=lambda item:item[1])
	avg=0
	for i in range(k):
		avg=avg+y_train[l[i][0]]
	avg=avg/k
	y_pred.append(avg)

mse=mean_squared_error(y_pred,y_test)
print(f"RMSE:{np.sqrt(mse)}")
"""X_train_100=X_train[100]
m=[]
for id,z in enumerate(X_train):
	d=distance(X_train_100,z)
	m.append((id,d))
	m.sort(key=lambda item:item[1])
a=0
for i in range(k):
	a=a+y_train[m[i][0]]
y_pred_100=a/k
print(f"Y_true:{y_train[100]} Y_pred:{y_pred_100}")
"""
