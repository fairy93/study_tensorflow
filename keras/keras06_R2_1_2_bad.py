# *과제1
#1. R2를 음수가 아닌 0.5 이하로 만들어라
#2. 데이터 건들지 마
#3. 레이어는 인풋 아웃풋 포함 6개 이상
#4 . batch_size=1
#5. epochs는 100이상
#6. 히든 레이어의 노드는 10개이상 1000개이하
#7. train70%

# *과제2

from sklearn.datasets import load_boston

datasets = load_boston()
x=datasets.data
y=datasets.target

print(x.shape)
print(y.shape)