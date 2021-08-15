import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x= datasets.data
y= datasets.target

pca = PCA(n_components=7)
x= pca
#2. 모델 

from xgboost 
