from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import mglearn
import numpy as np
X,y=mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
#we increase the default settings of "max_iter"
#lowering alpha allowed to make a more complex model
lasso=Lasso(alpha=0.01,max_iter=100000).fit(X_train,y_train)
print(f'Training score:{lasso.score(X_train,y_train)}')
print(f'Test score:{lasso.score(X_test,y_test)}')
print(f'Co-efficients used:{np.sum(lasso.coef_!=0)}')