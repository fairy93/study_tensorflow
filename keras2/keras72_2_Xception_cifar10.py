import time

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D,UpSampling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.applications import VGG16, VGG19,Xception
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
xception = Xception(weights='imagenet', include_top=False, input_shape=(96,96,3))

xception.trainable = True

model = Sequential()
model.add(UpSampling2D(size=(3,3)))
model.add(xception)
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
# xception.trainable = True /  Flatten
# fit time :  1378.0647513866425
# loss :  0.28042176365852356
# val_loss :  1.7903343439102173
# acc :  0.915066659450531
# val acc : 0.6362000107765198

# xception.trainable = False /  Flatten
# fit time :  272.54600954055786
# loss :  0.3161272704601288
# val_loss :  0.8918140530586243
# acc :  0.8861777782440186
# val acc : 0.746999979019165

#  ================== xception.trainable = True /  GlobalAveragePooling2D   =====best=========
# fit time :  1473.49041056633
# loss :  0.051613058894872665
# val_loss :  0.4055415987968445
# acc :  0.9843555688858032
# val acc : 0.90119999647140

# xception.trainable = False /  GlobalAveragePooling2D
# fit time :  272.491073846817
# loss :  0.1953636258840561
# val_loss :  1.001098871231079
# acc :  0.9307777881622314
# val acc : 0.7427999973297119


# ---------------------cifar100---------------------------
# xception.trainable = True /  Flatten
# fit time :  995.3748097419739
# loss :  4.605439186096191
# val_loss :  4.607487678527832
# acc :  0.008933333680033684
# val acc : 0.007400000002235174

# xception.trainable = False /  Flatten
# fit time :  270.7896966934204
# loss :  0.9821863770484924
# val_loss :  2.349381923675537
# acc :  0.7049777507781982
# val acc : 0.477400004863739

#  ================== xception.trainable = True /  GlobalAveragePooling2D   =====best=========
# fit time :  1474.5395603179932
# loss :  0.2018025666475296
# val_loss :  1.484706997871399
# acc :  0.9411110877990723
# val acc : 0.6773999929428101

# xception.trainable = False /  GlobalAveragePooling2D
# fit time :  252.23363327980042
# loss :  0.8068179488182068
# val_loss :  2.2080941200256348
# acc :  0.7568222284317017
# val acc : 0.49619999527931213