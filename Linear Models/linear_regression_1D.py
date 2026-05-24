from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mglearn
X,y=mglearn.datasets.make_wave(n_samples=60)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
lr=LinearRegression().fit(X_train,y_train)
print(f'lr_coeff:{lr.coef_}')
print(f'lr_intercept:{lr.intercept_}')
print(f'Training score:{lr.score(X_train,y_train)}')
print(f'Test score:{lr.score(X_test,y_test)}')
#As the R^2 score are low and almost similar ,it is a case of "underfitting"