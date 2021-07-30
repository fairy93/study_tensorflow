from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요',
        '별로에요','생각보다 지루해요','연기가 어색해요',
        '재미없어요','너무 재미없다', '참 재밋네요','청순이가 잘 생기긴 했어요'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token=Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

# x= token.texts_to_sequences(docs)
# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]
word_size = len(token.word_index)
print(word_size)  # 27

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM


model = Sequential()
model.add(Embedding(input_dim=28,output_dim=,i))