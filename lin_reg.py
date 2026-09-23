import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
epoch=1000
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
alpha=0.1
for i in range(epoch):
	y_pred=np.dot(X_train,W)+b
	error=(y_pred-y_train)
	dW=(2/n_samples)*np.dot(X_train.T,error)
	db=(2/n_samples)*np.sum(error)
	W=W-(alpha*dW)
	b=b-(alpha*db)
print(f"W:{W} b:{b}")
Y_pred=np.dot(X_test,W)+b
mse=mean_squared_error(y_test,Y_pred)
rmse=np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")
