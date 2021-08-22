import time

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D,UpSampling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.applications import VGG16, VGG19,Xception,ResNet101
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
res = ResNet101(weights='imagenet', include_top=False, input_shape=(32,32,3))

res.trainable = False

model = Sequential()
model.add(res)
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
# res.trainable = True /  Flatten
# fit time :  809.7662341594696
# loss :  0.9098713397979736
# val_loss :  1.0992385149002075
# acc :  0.6860222220420837
# val acc : 0.640999972820282

# res.trainable = False /  Flatten
# fit time :  1005.4653604030609
# loss :  1.7842249870300293
# val_loss :  1.7556990385055542
# acc :  0.3559555411338806
# val acc : 0.3725999891757965

#  ================== res.trainable = True /  GlobalAveragePooling2D   =====best=========
# fit time :  595.610523223877
# loss :  0.2201211303472519
# val_loss :  0.8774073123931885
# acc :  0.9282888770103455
# val acc : 0.7605999708175659

# res.trainable = False /  GlobalAveragePooling2D
# fit time :  1049.483582019806
# loss :  1.8055661916732788
# val_loss :  1.753343105316162
# acc :  0.34702223539352417
# val acc : 0.37299999594688416

# ---------------------cifar100---------------------------
# ==================  res.trainable = True /  Flatten  =====best=========
# fit time :  786.0149557590485
# loss :  1.0452827215194702
# val_loss :  2.7485885620117188
# acc :  0.6996666789054871
# val acc : 0.42480000853538513

# res.trainable = False /  Flatten
# fit time :  160.42136001586914
# loss :  4.6054463386535645
# val_loss :  4.607479095458984
# acc :  0.010177778080105782
# val acc : 0.007000000216066837

#  res.trainable = True /  GlobalAveragePooling2D 
# fit time :  605.2113373279572
# loss :  1.875963568687439
# val_loss :  2.576847553253174
# acc :  0.49451109766960144
# val acc : 0.3783999979496002

# res.trainable = False /  GlobalAveragePooling2D
# fit time :  1266.7590634822845
# loss :  3.83695387840271
# val_loss :  3.9488768577575684
# acc :  0.12008889019489288
# val acc : 0.11339999735355377