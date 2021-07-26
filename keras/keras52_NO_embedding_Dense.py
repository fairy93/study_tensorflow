from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요',
        '별로에요','생각보다 지루해요','연기가 어색해요',
        '재미없어요','너무 재미없다', '참 재밋네요','청순이가 잘 생기긴 했어요'
]
# 모델의 크기가 달라
# 긍정1, 부정 0

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])


token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x= token.texts_to_sequences(docs)
print(x)

# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15],
#  [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x,padding='pre',maxlen=5) # post
print(pad_x)
print(pad_x.shape) # 13.5

word_size = len(token.word_index)
print(word_size)  # 27

# # 원핫인코딩하면 머로바껴 13,5 -> 13,5,27

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
model.add(Dense(32,input_shape=(5,)))
model.add(Dense(16))
model.add(Dense(1,activation='sigmoid'))
#input dim = 라벨의갯수 단어사전의갯수
#input_length 5 백터화

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 5, 11)             297
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                5632
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33
# =================================================================
# Total params: 5,962
# Trainable params: 5,962
# Non-trainable params: 0
# _________________________________________________________________
#3. 컴파일,훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(pad_x,labels,epochs=100,batch_size=32)

# #4. 평가,예측wwwwwwwww
acc = model.evaluate(pad_x,labels)[1]
print('acc',acc)