from tensorflow.keras.preprocessing.text import Tokenizer
text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text]) # fit_on_texts() 메서드는 문자 데이터를 입력받아서 리스트의 형태로 변환한다

print(token.word_index) # {'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7}
# tokenizer의 word_index 속성은 단어와 숫자의 키-값 쌍을 포함하는 딕셔너리를 반환한다.
# 단어 빈도수가 높은 순으로 낮은 정수 인덱스를 부여하는데,

x = token.texts_to_sequences([text]) # 텍스트 안의 단어들을 숫자의 시퀀스의 형태로 변환한다.
# [[3, 1, 4, 5, 6, 1, 2, 2, 7]]
print(x)

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size) # 7
x = to_categorical(x)

print(x) 
print(x.shape)  # 1,9,8