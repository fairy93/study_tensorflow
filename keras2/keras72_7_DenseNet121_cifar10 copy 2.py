import time

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D,UpSampling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.applications import VGG16, VGG19,Xception,ResNet101,InceptionV3,DenseNet121
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
denseNet121 = DenseNet121(weights='imagenet', include_top=False, input_shape=(32,32,3))

denseNet121.trainable = False

model = Sequential()
model.add(denseNet121)
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



# ---------------------cifar10---------------------------
#  denseNet121.trainable = True /  Flatten
# fit time :  510.05545473098755
# loss :  0.11780949681997299
# val_loss :  0.7116531133651733
# acc :  0.9594444632530212
# val acc : 0.8086000084877014

# denseNet121.trainable = False /  Flatten
# fit time :  144.52350330352783
# loss :  0.6268920302391052
# val_loss :  1.0572474002838135
# acc :  0.7740444540977478
# val acc : 0.656000018119812

# ==================  denseNet121.trainable = True /  GlobalAveragePooling2D   =====best=========
# fit time :  663.6660008430481
# loss :  0.10238318145275116
# val_loss :  0.6667981147766113
# acc :  0.9655110836029053
# val acc : 0.8379999995231628

# denseNet121.trainable = False /  GlobalAveragePooling2D
# fit time :  167.4288158416748
# loss :  0.5406018495559692
# val_loss :  1.094980239868164
# acc :  0.8055777549743652
# val acc : 0.6582000255584717

# ---------------------cifar100---------------------------
# ==================  denseNet121.trainable = True /  Flatten  =====best=========
# fit time :  456.44577980041504
# loss :  0.6803157329559326
# val_loss :  2.032607316970825
# acc :  0.791266679763794
# val acc : 0.5393999814987183

# denseNet121.trainable = False /  Flatten
# fit time :  162.71334171295166
# loss :  1.4316879510879517
# val_loss :  2.61027455329895
# acc :  0.5924222469329834
# val acc : 0.37619999051094055

# denseNet121.trainable = True /  GlobalAveragePooling2D 
# fit time :  485.8883602619171
# loss :  0.712613582611084
# val_loss :  2.0352025032043457
# acc :  0.7812444567680359
# val acc : 0.5275999903678894

# denseNet121.trainable = False /  GlobalAveragePooling2D
# fit time :  158.67643451690674
# loss :  1.535799264907837
# val_loss :  2.541520357131958
# acc :  0.5662888884544373
# val acc : 0.3928000032901764