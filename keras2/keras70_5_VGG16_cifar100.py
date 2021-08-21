import time

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.applications import VGG16, VGG19
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
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

vgg16.trainable = False # or True

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=256,
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

#  ================== vgg16.trainable = True /  Flatten   =====best=========
# fit time :  224.62921595573425
# loss :  1.3963515758514404
# val_loss :  2.87082576751709
# acc :  0.5750444531440735
# val acc : 0.3601999878883362

# vgg16.trainable = False /  Flatten
# fit time :  68.77398014068604
# loss :  1.6791582107543945
# val_loss :  2.671828269958496
# acc :  0.5315777659416199
# val acc : 0.373199999332428

# vgg16.trainable = True /  GlobalAveragePooling2D
# fit time :  247.27679252624512
# loss :  1.5486295223236084
# val_loss :  2.9689548015594482
# acc :  0.5293333530426025
# val acc : 0.3149999976158142

# vgg16.trainable = False /  GlobalAveragePooling2D
# fit time :  71.70841073989868
# loss :  1.5864423513412476
# val_loss :  2.6314594745635986
# acc :  0.5575110912322998
# val acc : 0.3723999857902527
