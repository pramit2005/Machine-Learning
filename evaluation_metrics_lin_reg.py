import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
epoch=100000
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
n_samples,n_features=X_train.shape
W=np.zeros(n_features)
b=0
alpha=0.0001
for i in range(epoch):
	y_pred=np.dot(X_train,W)+b
	error=(y_pred-y_train)
	dW=(2/n_samples)*np.dot(X_train.T,error)
	db=(2/n_samples)*np.sum(error)
	W=W-(alpha*dW)
	b=b-(alpha*db)
Y_pred=np.dot(X_test,W)+b
mae=0
mse=0
x=0
y_mean=np.mean(y_test)
for i,j in zip(y_test,Y_pred):
	mae=mae+abs(i-j)
	mse=mse+((i-j)**2)
	x=x+(((i-j)**2)/((i-y_mean)**2))
mae=mae/len(Y_pred)
mse=mse/len(Y_pred)
r2=1-x
print(f"MAE:{mae}\nMSE:{mse}\nRMSE:{np.sqrt(mse)}\nR2 score:{r2}")

