import time

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data() # (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)
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
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

vgg16.trainable = True 

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

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


# vgg16.trainable = True /  Flatten
# fit time :  210.35920691490173
# loss :  0.264631986618042
# val_loss :  0.7848652601242065
# acc :  0.9151777625083923
# val acc : 0.7983999848365784

# vgg16.trainable = False /  Flatten
# fit time :  77.97564625740051
# loss :  0.7975121736526489
# val_loss :  1.0824192762374878
# acc :  0.7163333296775818
# val acc : 0.6402000188827515

#  ================== vgg16.trainable = True /  GlobalAveragePooling2D   =====best=========
# fit time :  224.2701165676117
# loss :  0.23891696333885193
# val_loss :  0.7397343516349792
# acc :  0.9266444444656372
# val acc : 0.8011999726295471

# vgg16.trainable = False /  GlobalAveragePooling2D
# fit time :  95.48902869224548
# loss :  0.7093015909194946
# val_loss :  1.1279646158218384
# acc :  0.7481111288070679
# val acc : 0.6358000040054321
