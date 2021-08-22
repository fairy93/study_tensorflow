import time

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D,UpSampling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.applications import VGG16, VGG19,Xception,ResNet50
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
res = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))

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
# fit time :  283.32691407203674
# loss :  0.5238248109817505
# val_loss :  0.8002877235412598
# acc :  0.829288899898529
# val acc : 0.76419997215271

# res.trainable = False /  Flatten
# fit time :  201.84939551353455
# loss :  1.766120433807373
# val_loss :  1.7227810621261597
# acc :  0.3585333228111267
# val acc : 0.384799987077713

#  ================== res.trainable = True /  GlobalAveragePooling2D   =====best=========
# fit time :  333.99097084999084
# loss :  0.1622912585735321
# val_loss :  1.5041964054107666
# acc :  0.9477555751800537
# val acc : 0.650600016117096

# res.trainable = False /  GlobalAveragePooling2D
# fit time :  288.7257778644562
# loss :  1.7490150928497314
# val_loss :  1.671687126159668
# acc :  0.36624443531036377
# val acc : 0.39340001344680786

# ---------------------cifar100---------------------------
# ==================  res.trainable = True /  Flatten  =====best=========
# fit time :  394.15064096450806
# loss :  0.6085043549537659
# val_loss :  3.203582286834717
# acc :  0.8181999921798706
# val acc : 0.4020000100135803

# res.trainable = False /  Flatten
# fit time :  84.43285942077637
# loss :  4.605489730834961
# val_loss :  4.607453346252441
# acc :  0.009355555288493633
# val acc : 0.007000000216066837

#  res.trainable = True /  GlobalAveragePooling2D 
# fit time :  299.90688014030457
# loss :  1.5730446577072144
# val_loss :  2.844817638397217
# acc :  0.5632666945457458
# val acc : 0.3720000088214874


# res.trainable = False /  GlobalAveragePooling2D
# fit time :  82.87015223503113
# loss :  4.605551242828369
# val_loss :  4.60743522644043
# acc :  0.009022221900522709
# val acc : 0.007400000002235174