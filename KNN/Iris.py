import pandas as pd
import mglearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
iris_dataset=load_iris()
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
#Next part shows the pair plot of the training data,showing how the features data are separated
#iris_df=pd.DataFrame(X_train,columns=iris_dataset.feature_names)
#a=pd.plotting.scatter_matrix(iris_df,c=y_train,figsize=(15,15),marker='0',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)
#plt.show()
knn=KNeighborsClassifier(n_neighbors=1) #taking the label of closest 1 data point
knn.fit(X_train,y_train)
X_new=np.array([[5,2.9,1,0.2]])
prediction=knn.predict(X_new)
print(f'Predicted target name:{prediction}')
print(f"The predicted target name:{iris_dataset['target_names'][prediction]}")
#checking accuracy
print(f'Test score:{knn.score(X_test,y_test):.2f}') #knn.score calculates the accuracy
#alternative way to calculate accuracy
y_pred=knn.predict(X_test)
print(f'Test score(Mean method):{np.mean(y_pred==y_test):.2f}')