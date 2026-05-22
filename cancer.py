from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
cancer=load_breast_cancer()
#print(f'The keys are{cancer.keys()}')
#print(f'The data:\n{cancer.feature_names}')
print("Sample counts per class:{}".format({n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}))