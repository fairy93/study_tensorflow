import time

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D,UpSampling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.applications import VGG16, VGG19,MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar100

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data() # (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)
x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train)
y_test = onehot.transform(y_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

#2. 모델
mobileNetV2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32,32,3))

mobileNetV2.trainable = False

model = Sequential()
model.add(mobileNetV2)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
                 verbose=2, validation_split=0.1, callbacks=[es])
end_time = time.time()-start_time

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('fit time : ', end_time)
print('loss : ', loss[-6])
print('val_loss : ', val_loss[-6])
print('acc : ', acc[-6])
print('val acc :', val_acc[-6])



# ---------------------cifar10---------------------------
#  mobileNetV2.trainable = True /  Flatten
# fit time :  234.0611093044281
# loss :  0.20313452184200287
# val_loss :  2.3628737926483154
# acc :  0.9327333569526672
# val acc : 0.673799991607666

# mobileNetV2.trainable = False /  Flatten
# fit time :  76.52037358283997
# loss :  1.5964477062225342
# val_loss :  1.7695695161819458
# acc :  0.42026665806770325
# val acc : 0.3614000082015991

# ==================  mobileNetV2.trainable = True /  GlobalAveragePooling2D   =====best=========
# fit time :  271.4131278991699
# loss :  0.16161121428012848
# val_loss :  1.7537453174591064
# acc :  0.946066677570343
# val acc : 0.6916000247001648

# mobileNetV2.trainable = False /  GlobalAveragePooling2D
# fit time :  77.55090999603271
# loss :  1.5968546867370605
# val_loss :  1.7760814428329468
# acc :  0.41984444856643677
# val acc : 0.3614000082015991



# ---------------------cifar100---------------------------
# ==================  mobileNetV2.trainable = True /  Flatten  =====best=========
# fit time :  280.50383734703064
# loss :  1.2580461502075195
# val_loss :  3.6642932891845703
# acc :  0.6394000053405762
# val acc : 0.31439998745918274

# mobileNetV2.trainable = False /  Flatten
# fit time :  81.08583211898804
# loss :  3.3300256729125977
# val_loss :  3.7221739292144775
# acc :  0.20640000700950623
# val acc : 0.1387999951839447

# mobileNetV2.trainable = True /  GlobalAveragePooling2D 
# fit time :  304.7545077800751
# loss :  0.9381453394889832
# val_loss :  3.278465509414673
# acc :  0.718999981880188
# val acc : 0.3977999985218048

# mobileNetV2.trainable = False /  GlobalAveragePooling2D
# fit time :  86.07567405700684
# loss :  3.332425117492676
# val_loss :  3.7377827167510986
# acc :  0.2049555629491806
# val acc : 0.1454000025987625