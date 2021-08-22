import time

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D,UpSampling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.applications import VGG16, VGG19,Xception,ResNet101,InceptionV3
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
inceptionv3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(96,96,3))

inceptionv3.trainable = False

model = Sequential()
model.add(UpSampling2D(size=(3,3)))
model.add(inceptionv3)
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
# ==================  inceptionv3.trainable = True /  Flatten  =====best=========
# fit time :  712.6318626403809
# loss :  0.07325447350740433
# val_loss :  0.6057670712471008
# acc :  0.9762444496154785
# val acc : 0.8547999858856201


# inceptionv3.trainable = False /  Flatten
# fit time :  165.8871455192566
# loss :  0.4655179977416992
# val_loss :  1.0728111267089844
# acc :  0.8341110944747925
# val acc : 0.6747999787330627

# inceptionv3.trainable = True /  GlobalAveragePooling2D 
# fit time :  602.8879747390747
# loss :  0.09447260946035385
# val_loss :  0.6736450791358948
# acc :  0.9698888659477234
# val acc : 0.8241999745368958

# inceptionv3.trainable = False /  GlobalAveragePooling2D
# fit time :  166.8816237449646
# loss :  0.4275730848312378
# val_loss :  1.1200369596481323
# acc :  0.8462222218513489
# val acc : 0.678600013256073

# ---------------------cifar100---------------------------
#  inceptionv3.trainable = True /  Flatten
# fit time :  1003.8905982971191
# loss :  0.6187460422515869
# val_loss :  3.362488269805908
# acc :  0.8195777535438538
# val acc : 0.38659998774528503

# inceptionv3.trainable = False /  Flatten
# fit time :  163.412260055542
# loss :  1.3372911214828491
# val_loss :  2.513248920440674
# acc :  0.6116889119148254
# val acc : 0.3984000086784363

# ==================  inceptionv3.trainable = True /  GlobalAveragePooling2D   =====best=========

# fit time :  757.7810418605804
# loss :  0.45299211144447327
# val_loss :  2.288175582885742
# acc :  0.8617555499076843
# val acc : 0.5343999862670898

# inceptionv3.trainable = False /  GlobalAveragePooling2D
# fit time :  166.7958378791809
# loss :  1.333748459815979
# val_loss :  2.6420655250549316
# acc :  0.6139110922813416
# val acc : 0.39079999923706055