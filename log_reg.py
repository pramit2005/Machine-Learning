import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
def sigmoid(z):
	return 1 / (1 + np.exp(-z))
epoch=10000
df=pd.read_csv('credit_card.csv')
df=df.dropna()
df=df.drop_duplicates()
df_y=df['Fraud']
df_X=df.drop(columns=['Fraud'])
X=df_X.to_numpy()
y=df_y.to_numpy()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
n_samples,n_features=X_train.shape
W=np.zeros(n_features)
b=0
alpha=1
for i in range(epoch):
	z=np.dot(X_train,W)+b
	y_pred=sigmoid(z)
	error=y_pred-y_train
	dW=(1/n_samples)*np.dot(X_train.T,error)
	db=(1/n_samples)*np.sum(error)
	W=W-(alpha*dW)
	b=b-(alpha*db)
Y_pred=[]
for x in X_test:
	z=x@W+b
	y_pred=sigmoid(z)
	if y_pred>=0.5:
		Y_pred.append(1)
	else:
		Y_pred.append(0)
print(f"Accuracy:{accuracy_score(y_test,Y_pred)}")
