import time

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D,UpSampling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.applications import VGG16, VGG19,EfficientNetB0
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
eff = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32,32,3))

eff.trainable = False

model = Sequential()
# model.add(UpSampling2D(size=(7,7)))
model.add(eff)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
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
# fit time :  313.87414479255676
# loss :  0.3016566336154938
# val_loss :  6.168700218200684
# acc :  0.8990222215652466
# val acc : 0.10279999673366547

# mobileNetV2.trainable = False /  Flatten
# fit time :  106.27515745162964
# loss :  2.3028464317321777
# val_loss :  2.302849292755127
# acc :  0.10215555876493454
# val acc : 0.09759999811649323

# ==================  mobileNetV2.trainable = True /  GlobalAveragePooling2D   =====best=========
# fit time :  300.28212332725525
# loss :  0.34113988280296326
# val_loss :  3.837052822113037
# acc :  0.8836444616317749
# val acc : 0.12439999729394913

# mobileNetV2.trainable = False /  GlobalAveragePooling2D
# fit time :  116.45101022720337
# loss :  2.3027396202087402
# val_loss :  2.302748918533325
# acc :  0.09806666523218155
# val acc : 0.09700000286102295


# ---------------------cifar100---------------------------
# ==================  mobileNetV2.trainable = True /  Flatten  =====best=========
# fit time :  323.6974296569824
# loss :  1.061908483505249
# val_loss :  9.26948070526123
# acc :  0.6906889081001282
# val acc : 0.01720000058412552

# mobileNetV2.trainable = False /  Flatten


# mobileNetV2.trainable = True /  GlobalAveragePooling2D 


# mobileNetV2.trainable = False /  GlobalAveragePooling2D
