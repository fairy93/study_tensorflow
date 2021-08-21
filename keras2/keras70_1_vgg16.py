from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16,VGG19

# imagenet
#   1,000개의 클래스로 구성되며 총 백만 개가 넘는 데이터를 포함
#   약 120만 개는 학습(training) 5만개는 검증(validation)에
#   학습 데이터셋 용량은 약 138GB, 검증 데이터셋 용량은 약 6GB
#   학습 데이터를 확인해 보면 각 클래스당 약 1,000개가량의 사진으로 구성


# default false
# include_top = 위 아래 커스터마이징
model = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))
model.trainable=False # default True
# model = VGG16()
# model = VGG19()


model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

# input_1 (InputLayer)         [(None, 224, 224, 3)]     0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
# ............................
# ............................
# flatten (Flatten)            (None, 25088)             0
# _________________________________________________________________
# fc1 (Dense)                  (None, 4096)              102764544
# _________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312
# _________________________________________________________________
# predictions (Dense)          (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0

# ​
# FC(Fully connected layer)

# 완전히 연결 되었다라는 뜻으로,
# 한층의 모든 뉴런이 다음층이 모든 뉴런과 연결된 상태로
# 2차원의 배열 형태 이미지를 1차원의 평탄화 작업을 통해 이미지를 분류하는데 사용되는 계층입니다.
# 1. 2차원 배열 형태의 이미지를 1차원 배열로 평탄화
# 2. 활성화 함수(Relu, Leaky Relu, Tanh,등)뉴런을 활성화
# 3. 분류기(Softmax) 함수로 분류