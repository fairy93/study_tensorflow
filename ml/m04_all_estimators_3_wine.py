import warnings
from numpy.lib.npyio import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_wine
from sklearn import datasets

warnings.filterwarnings('ignore')

#1 데이터
datasets = load_wine()

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
# AdaBoostClassifier() acc :  0.9444444444444444
# BaggingClassifier() acc :  0.9629629629629629
# BernoulliNB() acc :  0.4444444444444444
# CalibratedClassifierCV() acc :  0.9444444444444444
# CategoricalNB not found
# ClassifierChain not found
# ComplementNB() acc :  0.6481481481481481
# DecisionTreeClassifier() acc :  0.9259259259259259
# DummyClassifier() acc :  0.4444444444444444
# ExtraTreeClassifier() acc :  0.8888888888888888
# ExtraTreesClassifier() acc :  1.0
# GaussianNB() acc :  1.0
# GaussianProcessClassifier() acc :  0.5185185185185185
# GradientBoostingClassifier() acc :  1.0
# HistGradientBoostingClassifier() acc :  0.9629629629629629
# KNeighborsClassifier() acc :  0.7222222222222222
# LabelPropagation() acc :  0.42592592592592593
# LabelSpreading() acc :  0.42592592592592593
# LinearDiscriminantAnalysis() acc :  1.0
# LinearSVC() acc :  0.9814814814814815
# LogisticRegression() acc :  0.9629629629629629
# LogisticRegressionCV() acc :  0.9629629629629629
# MLPClassifier() acc :  0.9444444444444444
# MultiOutputClassifier not found
# MultinomialNB() acc :  0.9074074074074074
# NearestCentroid() acc :  0.7222222222222222
# NuSVC() acc :  0.8333333333333334
# OneVsOneClassifier not found
# OneVsRestClassifier not found
# OutputCodeClassifier not found
# PassiveAggressiveClassifier() acc :  0.6666666666666666
# Perceptron() acc :  0.5370370370370371
# QuadraticDiscriminantAnalysis() acc :  0.9814814814814815
# RadiusNeighborsClassifier not found
# RandomForestClassifier() acc :  1.0
# RidgeClassifier() acc :  0.9814814814814815
# RidgeClassifierCV(alphas=array([ 0.1,  1. , 10. ])) acc :  0.9814814814814815
# SGDClassifier() acc :  0.5185185185185185
# SVC() acc :  0.6481481481481481
# StackingClassifier not found
# VotingClassifier not found