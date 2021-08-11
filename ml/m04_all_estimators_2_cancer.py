import warnings
from numpy.lib.npyio import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_breast_cancer
from sklearn import datasets

warnings.filterwarnings('ignore')

#1 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=79)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier') #분류 (type_filter ='regressor')
# print(allAlgorithms) 
print('모델의 갯수',len(allAlgorithms))
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train,y_train)

        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test,y_predict)
        print(model,'acc : ',acc)
    except:
        print(name,'not found')
        continue

# 모델의 갯수 41
# AdaBoostClassifier() acc :  0.9532163742690059
# BaggingClassifier() acc :  0.9532163742690059
# BernoulliNB() acc :  0.6491228070175439
# CalibratedClassifierCV() acc :  0.9181286549707602
# CategoricalNB not found
# ClassifierChain not found
# ComplementNB() acc :  0.8947368421052632
# DecisionTreeClassifier() acc :  0.9181286549707602
# DummyClassifier() acc :  0.6491228070175439
# ExtraTreeClassifier() acc :  0.9181286549707602
# ExtraTreesClassifier() acc :  0.9590643274853801
# GaussianNB() acc :  0.9415204678362573
# GaussianProcessClassifier() acc :  0.9122807017543859
# GradientBoostingClassifier() acc :  0.9532163742690059
# HistGradientBoostingClassifier() acc :  0.9649122807017544
# KNeighborsClassifier() acc :  0.9298245614035088
# LabelPropagation() acc :  0.3684210526315789
# LabelSpreading() acc :  0.3684210526315789
# LinearDiscriminantAnalysis() acc :  0.9532163742690059
# LinearSVC() acc :  0.9298245614035088
# LogisticRegression() acc :  0.935672514619883
# LogisticRegressionCV() acc :  0.9590643274853801
# MLPClassifier() acc :  0.9415204678362573
# MultiOutputClassifier not found
# MultinomialNB() acc :  0.8947368421052632
# NearestCentroid() acc :  0.9005847953216374
# NuSVC() acc :  0.8888888888888888
# OneVsOneClassifier not found
# OneVsRestClassifier not found
# OutputCodeClassifier not found
# PassiveAggressiveClassifier() acc :  0.8947368421052632
# Perceptron() acc :  0.5906432748538012
# QuadraticDiscriminantAnalysis() acc :  0.9239766081871345
# RadiusNeighborsClassifier not found
# RandomForestClassifier() acc :  0.9590643274853801
# RidgeClassifier() acc :  0.9415204678362573
# RidgeClassifierCV(alphas=array([ 0.1,  1. , 10. ])) acc :  0.9590643274853801
# SGDClassifier() acc :  0.8713450292397661
# SVC() acc :  0.9181286549707602
# StackingClassifier not found
# VotingClassifier not found