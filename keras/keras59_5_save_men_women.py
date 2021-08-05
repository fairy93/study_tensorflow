import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/men_women',
    target_size=(150, 150),
    batch_size=2000,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

# xy_test = test_datagen.flow_from_directory(
#     '../_data/men_women',
#     target_size=(150,150),
#     batch_size=3000,
#     class_mode='binary',
#     shuffle=True,
#     subset='validation'
# )

x_pred = train_datagen.flow_from_directory(
    '../_data/kdj',
    target_size=(150, 150),
    batch_size=2000,
    class_mode='binary',
)

np.save('./_save/_npy/k59_5_train_x.npy', arr=xy_train[0][0])
np.save('./_save/_npy/k59_5_train_y.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/k59_5_test_x.npy',arr=xy_test[0][0])
# np.save('./_save/_npy/k59_5_test_y.npy',arr=xy_test[0][1])
np.save('./_save/_npy/k59_5_pred_x.npy', arr=x_pred[0][0])
