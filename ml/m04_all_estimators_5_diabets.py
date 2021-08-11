import warnings
from numpy.lib.npyio import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_diabetes
from sklearn import datasets

warnings.filterwarnings('ignore')

#1 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=79)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='regressor') #분류 (type_filter ='regressor')
# print(allAlgorithms) 
print('모델의 갯수',len(allAlgorithms))
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train,y_train)

        y_predict = model.predict(x_test)
        r2 = r2_score(y_test,y_predict)
        print(model,'r2 : ',r2)
    except:
        print(name,'not found')
        continue

# 모델의 갯수 54
# ARDRegression() r2 :  0.4308105028430459
# AdaBoostRegressor() r2 :  0.38091394349037033
# BaggingRegressor() r2 :  0.3924786065418072      
# BayesianRidge() r2 :  0.44081293374217534        
# CCA() r2 :  0.3224720855466512
# DecisionTreeRegressor() r2 :  0.08881304419634717
# DummyRegressor() r2 :  -0.009799156402168663     
# ElasticNet() r2 :  -0.0018642621917752678        
# ElasticNetCV() r2 :  0.4030755056877495       
# ExtraTreeRegressor() r2 :  -0.1330668654428142
# ExtraTreesRegressor() r2 :  0.4185518313709282      
# GammaRegressor() r2 :  -0.0038242347772659002       
# GaussianProcessRegressor() r2 :  -10.113704210713752
# GradientBoostingRegressor() r2 :  0.3777749537082613
# HistGradientBoostingRegressor() r2 :  0.3882869731150398
# HuberRegressor() r2 :  0.43166858145881637
# IsotonicRegression not found
# KNeighborsRegressor() r2 :  0.3333959730299869
# KernelRidge() r2 :  -3.076697116464392
# Lars() r2 :  0.44040657403728
# LarsCV() r2 :  0.43620118231272065
# Lasso() r2 :  0.3132478314078804
# LassoCV() r2 :  0.4358514005714552
# LassoLars() r2 :  0.3442694668705717
# LassoLarsCV() r2 :  0.43579780268910817
# LassoLarsIC() r2 :  0.42476533279452966
# LinearRegression() r2 :  0.44040657403728
# LinearSVR() r2 :  -0.2348300803472334
# MLPRegressor() r2 :  -2.7157221859358978
# MultiOutputRegressor not found
# MultiTaskElasticNet not found
# MultiTaskElasticNetCV not found
# MultiTaskLasso not found
# MultiTaskLassoCV not found
# NuSVR() r2 :  0.13493042641312736
# OrthogonalMatchingPursuit() r2 :  0.16339396398536588
# OrthogonalMatchingPursuitCV() r2 :  0.4355135931212962
# PLSCanonical() r2 :  -0.9400417038869779
# PLSRegression() r2 :  0.4663295707858579
# PassiveAggressiveRegressor() r2 :  0.46136455551908595
# PoissonRegressor() r2 :  0.3032819393319428
# RANSACRegressor() r2 :  -0.5532219619262408
# RadiusNeighborsRegressor() r2 :  -0.009799156402168663
# RandomForestRegressor() r2 :  0.40891511454817775
# RegressorChain not found
# Ridge() r2 :  0.37118812434891957
# RidgeCV(alphas=array([ 0.1,  1. , 10. ])) r2 :  0.4487038157859875
# SGDRegressor() r2 :  0.34705132172838493
# SVR() r2 :  0.15833350821867287
# StackingRegressor not found
# TheilSenRegressor(max_subpopulation=10000) r2 :  0.41877364921565186
# TransformedTargetRegressor() r2 :  0.44040657403728
# TweedieRegressor() r2 :  -0.004150277635454502
# VotingRegressor not found