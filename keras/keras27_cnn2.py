from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D

model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2),   #(N,10,10,1) 
    padding='same', input_shape=(10,10,1)))   #(N,10,10,10)            이미지 통과시켜도 4차원 #paddig same 하면 5.5.10 으로전달
model.add(Conv2D(20,(2,2), activation='relu')) # (n,9,9,20)           # padding defalut = valid
model.add(Conv2D(30,(2,2), padding='valid')) # (n,8,8,30)                                          
model.add(MaxPooling2D())                     #(n,4,4,30)
model.add(Conv2D(15,(2,2)))                   #(n,3,3,15)
model.add(Flatten())                          #(N, 135) 여
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1,activation='sigmoid'))
model.summary()


# #rhkwp1 con2d 디폴트 액티비
# 과제2 컨2ㅇ summary 파라미터갯수 완벽이해
# 패딩 맥스폴링은쓰고싶을때 데이터처리할떄 인공지능은 한땀한땀하는거야
# 101010
# 9920
# 8830
# 4430
# 400