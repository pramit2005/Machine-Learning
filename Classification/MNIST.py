from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
mnist=fetch_openml('mnist_784',as_frame=False)
sgd_clf=SGDClassifier(random_state=42)
X,y=mnist.data,mnist.target
X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]
#we will test for only one digit '5' first
y_test_5=(y_test=='5')
y_train_5=(y_train=='5')
y_train_pred=cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)
#this is just like cross_val_score(), but it returns prediction made on each test fold,
# means we get "clean" prediction for each instance in the training set
print(f1_score(y_train_5,y_train_pred))
