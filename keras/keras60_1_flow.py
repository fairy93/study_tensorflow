import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,  # False
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,  # 10
    zoom_range=1.2,   # 0.1
    shear_range=0.7,  # 0.5
    fill_mode='nearest'
)
# train_datagen = ImageDataGenerator(rescale=1./255)

# xy_train = train_datagen.flow_from_directory(
#     '../_data/brain/train',
#     target_size=(150,150),
#     batch_size=5,
#     class_mode='binary'
# )
#1. ImageDataGenerator를 정의
#2. 파일에서 땡겨올려면 -> flow_from_directory() //x,y가 튜플형태로 뭉쳐있어
#3. 데이터에서 땡겨올려면 -> flow()                 // x,y가 나눠있어
argument_size = 100
x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28),
            argument_size).reshape(-1, 28, 28, 1),   # x
    np.zeros(argument_size),   # y
    batch_size=argument_size,
    shuffle=False
).next()  # iterator 방식으로 반환!! .netx() 후 해봐 밑에 print 들

# <class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
print(type(x_data))
print(type(x_data[0]))  # <class 'tuple'>
print(type(x_data[0][0]))  # <class 'numpy.ndarray'>
print(x_data[0][0].shape)  # (100, 28, 28, 1) -> x값
print(x_data[0][1].shape)  # (100,)

plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')
plt.show()
