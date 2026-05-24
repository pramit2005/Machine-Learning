from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mglearn
X,y=mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
reg=LinearRegression().fit(X_train,y_train)
print(f'Training score:{reg.score(X_train,y_train)}')
print(f'Test score:{reg.score(X_test,y_test)}')
#Overfitting as Training score is a lot more than test score