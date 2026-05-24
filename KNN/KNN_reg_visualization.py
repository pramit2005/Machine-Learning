from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import mglearn
import numpy as np
import matplotlib.pyplot as plt
X,y=mglearn.datasets.make_wave(n_samples=40)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
fig,axes=plt.subplots(1,3,figsize=(15,4))
line=np.linspace(-3,3,1000).reshape(-1,1)
for n_neighbors,ax in zip([1,3,9],axes):
    reg=KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train,y_train)
    ax.plot(line,reg.predict(line))
    ax.plot(X_train,y_train,'^',c=mglearn.cm2(0),markersize=8)
    ax.plot(X_test,y_test,'v',c=mglearn.cm2(1),markersize=8)
    ax.set_title(f'{n_neighbors} neighbors(s)\n train score: {reg.score(X_train,y_train):.2f} test score: {reg.score(X_test,y_test):.2f}')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
axes[0].legend(['Model Predictions','Training data/target','Test data/target'],loc='best')
plt.show()
