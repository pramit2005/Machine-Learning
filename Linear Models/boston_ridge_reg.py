from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import mglearn
X,y=mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
#alpha allows user to control the value of "w"
#Increasing alpha means the values of "w" moves more toward 0
#Decresing alpha means making coefficients(w) less restricted
#Default alpha value is 1
ridge=Ridge(alpha=0.1).fit(X_train,y_train)
print(f'Training score:{ridge.score(X_train,y_train)}')
print(f'Test score:{ridge.score(X_test,y_test)}')