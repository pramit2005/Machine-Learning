import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import mglearn
fig,axes=plt.subplots(1,3,figsize=(10,3))
X,y=mglearn.datasets.make_forge()
for n_neighbors,ax in zip([1,3,9],axes):
    clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=True,eps=0.5,ax=ax,alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title(f'{n_neighbors} neighbor(s)')
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
axes[0].legend(loc=3)
plt.show()
