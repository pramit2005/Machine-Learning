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
k = 7
def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

y_pred = []

for x in X_test:
    l = []

    for id, z in enumerate(X_train):
        d = distance(x, z)
        l.append((id, d))

    l.sort(key=lambda item: item[1])
    m = l[:k]
    neighbors = [y_train[item[0]] for item in m]

    x_1 = neighbors.count(1)
    x_0 = neighbors.count(0)

    if x_1 > x_0:
        y_pred.append(1)
    else:
        y_pred.append(0)

print(f"Accuracy: {accuracy_score(y_pred, y_test)*100}")

