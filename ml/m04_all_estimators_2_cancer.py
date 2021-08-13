import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=79)

#2. 모델
allAlgorithms = all_estimators(type_filter='classifier') # 분류 / # 회귀 (type_filter ='regressor')
# print(allAlgorithms) 
print('모델의 갯수',len(allAlgorithms))
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train,y_train)

        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(model,'acc : ',acc)
    except:
        print(name,'not found')
        continue

# 모델의모델의 갯수 41
# AdaBoostClassifier() acc :  0.9736842105263158
# BaggingClassifier() acc :  0.9649122807017544
# BernoulliNB() acc :  0.6578947368421053      
# CalibratedClassifierCV() acc :  0.9210526315789473
# CategoricalNB not found
# ClassifierChain not found
# ComplementNB() acc :  0.8947368421052632
# DecisionTreeClassifier() acc :  0.9035087719298246
# DummyClassifier() acc :  0.6578947368421053
# ExtraTreeClassifier() acc :  0.9473684210526315
# ExtraTreesClassifier() acc :  0.9736842105263158
# GaussianNB() acc :  0.9649122807017544
# GaussianProcessClassifier() acc :  0.9210526315789473
# GradientBoostingClassifier() acc :  0.9649122807017544
# HistGradientBoostingClassifier() acc :  0.9736842105263158
# KNeighborsClassifier() acc :  0.9385964912280702
# LabelPropagation() acc :  0.3684210526315789
# LabelSpreading() acc :  0.3684210526315789
# LinearDiscriminantAnalysis() acc :  0.9473684210526315
# LinearSVC() acc :  0.9473684210526315
# LogisticRegression() acc :  0.956140350877193
# LogisticRegressionCV() acc :  0.9649122807017544
# MLPClassifier() acc :  0.9385964912280702
# MultiOutputClassifier not found
# MultinomialNB() acc :  0.8947368421052632
# NearestCentroid() acc :  0.8947368421052632
# NuSVC() acc :  0.8947368421052632
# OneVsOneClassifier not found
# OneVsRestClassifier not found
# OutputCodeClassifier not found
# PassiveAggressiveClassifier() acc :  0.8508771929824561
# Perceptron() acc :  0.9210526315789473
# QuadraticDiscriminantAnalysis() acc :  0.9298245614035088
# RadiusNeighborsClassifier not found
# RandomForestClassifier() acc :  0.956140350877193
# RidgeClassifier() acc :  0.956140350877193
# RidgeClassifierCV(alphas=array([ 0.1,  1. , 10. ])) acc :  0.9649122807017544
# SGDClassifier() acc :  0.9298245614035088
# SVC() acc :  0.9210526315789473
# StackingClassifier not found
# VotingClassifier not found