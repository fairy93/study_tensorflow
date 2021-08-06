import keras
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
num_classes = 7  # angry, disgust, fear, happy, sad, surprise, neutral

#1 데이터
with open("face/data/data.csv") as f:
    content = f.readlines()

lines = np.array(content)
num_of_instances = lines.size

x_train, y_train, x_test, y_test = [], [], [], []

for i in range(1, num_of_instances):
    try:
        emotion, img, usage = lines[i].split(",")
        val = img.split(" ")  # type(val) <class 'list'>
        pixels = np.array(val, 'float32')

        emotion = to_categorical(emotion, num_classes)
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PrivateTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
	    pass

x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')

x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')


x_train /= 255 #normalize inputs between [0, 1]
x_test /= 255
# print(x_train.shape, y_train.shape) # (31851, 2304) (31851, 7)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.5,
    fill_mode='nearest'
)

augument_size = 30000
randix = np.random.randint(x_train.shape[0], size=augument_size)
x_augmented = x_train[randix].copy()
y_augmented = y_train[randix].copy()
# print(x_augmented.shape)  # (30000, 2304)

x_augmented = x_augmented.reshape(30000, 48, 48, 1)
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

x_augumented = train_datagen.flow(x_augmented, np.zeros(augument_size),
                                  batch_size=augument_size, shuffle=False).next()[0]   # x만 쏙빠진다 flow 4차원 받아야해


# (31851, 2304) (31851, 7)
x_train = np.concatenate((x_train, x_augmented))  # (100000, 28, 28, 1)
y_train = np.concatenate((y_train, y_augmented))  # (100000,)

x_train = x_train.reshape(x_train.shape[0], 48,48,1)  # 31851 train samples
x_test = x_test.reshape(x_test.shape[0], 48,48,1)  # 4371 test samples
x_train,x_val ,y_train, y_val = train_test_split(x_train,y_train, train_size=0.8,shuffle=True)
# x_train, x_val,y_train,y_val (25480, 48, 48, 1) (6371, 48, 48, 1) (25480, 7) (6371, 7)
print(x_train.shape, x_test.shape,y_train.shape)
# #2 모델 구성
model = Sequential()

model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# gen = ImageDataGenerator()
# train_generator = gen.flow(x_train, y_train, batch_size=128) # x, y 가져와 batch_size만큼 증가시킴 데이터

es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# hist = model.fit(x_train,y_train,epochs=1000, batch_size=256, validation_split=0.2, callbacks=[es],verbose=2)
hist = model.fit(x_train,y_train,batch_size=256,epochs=1000,validation_data=(x_val,y_val),callbacks=[es], verbose=2 )
# hist = model.fit_generator(train_generator, epochs=1000, steps_per_epoch=12, validation_data=(x_val,y_val),callbacks=[es], verbose=2)


train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])

test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])


def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')

    plt.show()

y_vloss = hist.history['val_loss']
y_loss = hist.history['loss']

y_vacc = hist.history['val_acc']
y_acc = hist.history['acc']


x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# # acc
# x_len = np.arange(len(y_acc))
# plt.plot(x_len, y_vacc, marker='.', c='red', label="Validation-set acc")
# plt.plot(x_len, y_acc, marker='.', c='blue', label="Train-set acc")

# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.show()

import cv2
img = image.load_img("face/tmp2.jpg", grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.imshow(x)
plt.show()

