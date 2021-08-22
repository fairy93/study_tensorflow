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
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

vgg19.trainable = False

model = Sequential()
model.add(vgg19)
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
# vgg19.trainable = True /  Flatten
# fit time :  375.9653432369232
# loss :  0.2968679368495941
# val_loss :  0.7944555282592773
# acc :  0.9074666500091553
# val acc : 0.7838000059127808

# vgg19.trainable = False /  Flatten
# fit time :  98.05996680259705
# loss :  0.8660882711410522
# val_loss :  1.1053714752197266
# acc :  0.6928444504737854
# val acc : 0.6164000034332275

#  ================== vgg19.trainable = True /  GlobalAveragePooling2D   =====best=========
# fit time :  423.1023826599121
# loss :  0.2658096253871918
# val_loss :  0.8318741321563721
# acc :  0.9200666546821594
# val acc : 0.7943999767303467

# vgg19.trainable = False /  GlobalAveragePooling2D
# fit time :  102.54475927352905
# loss :  0.8547269701957703
# val_loss :  1.1420307159423828
# acc :  0.6953333616256714
# val acc : 0.6123999953269958

# ---------------------cifr100---------------------------
# vgg19.trainable = True /  Flatten
# fit time :  420.2188632488251
# loss :  1.852311134338379
# val_loss :  3.0397603511810303
# acc :  0.45268890261650085
# val acc : 0.2962000072002411

# vgg19.trainable = False /  Flatten
# fit time :  107.40755271911621
# loss :  2.0503692626953125
# val_loss :  2.694114923477173
# acc :  0.4507555663585663
# val acc : 0.3434000015258789

# vgg19.trainable = True /  GlobalAveragePooling2D
# fit time :  392.0875086784363
# loss :  2.0970778465270996
# val_loss :  3.0106635093688965
# acc :  0.40104445815086365
# val acc : 0.2863999903202057

# ================== vgg19.trainable = False /  GlobalAveragePooling2D   =====best=========
# fit time :  112.991934299469
# loss :  2.010613441467285
# val_loss :  2.7241108417510986
# acc :  0.46086665987968445
# val acc : 0.3314000070095062