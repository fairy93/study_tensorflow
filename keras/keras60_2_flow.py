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
augument_size = 40000

randix = np.random.randint(x_train.shape[0], size=augument_size)
print(x_train.shape[0])  # 60000
print(randix)  # [25775  3849 40525 ...  5198 35278 51604]
print(randix.shape)  # (40000,)

x_augmented = x_train[randix].copy()
y_augmented = y_train[randix].copy()
print(x_augmented.shape)  # (40000, 28, 28)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_augumented = train_datagen.flow(x_augmented, np.zeros(augument_size),
                                  batch_size=augument_size, shuffle=False).next()[0]   # x만 쏙빠진다 flow 4차원 받아야해

print(x_augmented.shape)   # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))  # (100000, 28, 28, 1)
y_train = np.concatenate((y_train, y_augmented))  # (100000,)
print(x_train.shape, y_train.shape)
