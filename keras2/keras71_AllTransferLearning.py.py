from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet50,ResNet50V2
from tensorflow.keras.applications import ResNet101,ResNet101V2,ResNet152,ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169,DenseNet201
from tensorflow.keras.applications import InceptionV3,InceptionResNetV2
from tensorflow.keras.applications import MobileNet,MobileNetV2,MobileNetV3Large,MobileNetV3Small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1,EfficientNetB7

model = VGG16()
# Total params: 138,357,544
# total model :  32
# train model weights : 0

# model = VGG19
# Total params: 143,667,240
# total model :  38
# train model weights : 0

# model = Xception()
# Total params: 22,910,480
# total model :  236
# train model weights : 0

# model=ResNet50()
# Total params: 25,636,712
# total model :  320
# train model weights : 0

# model = ResNet50V2()
# Total params: 25,613,800
# total model :  272
# train model weights : 0

# model = ResNet101()
# Total params: 44,707,176
# total model :  626
# train model weights : 0

# model = ResNet101V2()
# Total params: 44,675,560
# total model :  544
# train model weights : 0

# model = ResNet152()
# Total params: 60,419,944
# total model :  932
# train model weights : 0

# model = ResNet152V2()
# Total params: 60,380,648
# total model :  816
# train model weights : 0

# model = DenseNet121()
# Total params: 8,062,504
# total model :  606
# train model weights : 0

# model = DenseNet169()
# Total params: 14,307,880
# total model :  846
# train model weights : 0

# model = DenseNet201()
# Total params: 20,242,984
# total model :  1006
# train model weights : 0

# model = InceptionV3()
# Total params: 23,851,784
# total model :  378
# train model weights : 0

# model = InceptionResNetV2()
# Total params: 55,873,736
# total model :  898
# train model weights : 0

# model = MobileNet()
# Total params: 4,253,864
# total model :  137
# train model weights : 0

# model = MobileNetV2()
# Total params: 3,538,984
# total model :  262
# train model weights : 0

# model = MobileNetV3Large()
# Total params: 5,507,432
# total model :  266
# train model weights : 0

# model = MobileNetV3Small()
# Total params: 2,554,968
# total model :  210
# train model weights : 0

# model = NASNetLarge()
# Total params: 88,949,818
# total model :  1546
# train model weights : 0

# model = NASNetMobile()
# Total params: 5,326,716
# total model :  1126
# train model weights : 0

# model = EfficientNetB0()
# Total params: 5,330,571
# total model :  314
# train model weights : 0

# model = EfficientNetB1()
# Total params: 7,856,239
# total model :  442
# train model weights : 0

# model = EfficientNetB7()
# Total params: 66,658,687
# total model :  1040
# train model weights : 0

model.trainable=False
model.summary()

print('total model : ',len(model.weights))
print('train model weights : ',model.trainable_weights)


#모델별로 파라미터와 웨이트 수들 정리할것