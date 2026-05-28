import keras.utils
import numpy as np
from keras import layers
from keras import Input
from keras.models import Model

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape=(None,), dtype='int32', name='text_input')
embedding_text = layers.Embedding(text_vocabulary_size, 64)(text_input)    # 将输入文本嵌入长度为64的向量
encoded_text = layers.LSTM(32)(embedding_text)   # 利用 LSTM 将向量编码为单个向量

question_input = Input(shape=(None,), dtype='int32', name='question_input')
embedding_question = layers.Embedding(question_vocabulary_size, 64)(question_input)
encoded_question = layers.LSTM(32)(embedding_question)

'''特征拼接（Concatenate）
在 Python 和深度学习中，axis=-1 代表“最后一个维度”（也就是最内层的特征维度）。'''
concatenated = layers.Concatenate(axis=-1)([encoded_question, encoded_text])

answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

model = Model(inputs=[text_input, question_input], outputs=answer)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])

num_samples = 100
max_length = 100

text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))  # 生成虚构的 Numpy 数据
answers = np.random.randint(answer_vocabulary_size, size=(num_samples,))
answers = keras.utils.to_categorical(answers, answer_vocabulary_size)  # 回答是 one—hot编码

# model.fit([text, question], answers, epochs=10, batch_size=128)   # 使用输入组成的列表拟合
model.fit({'text_input': text, 'question_input': question},
          answers, epochs=10, batch_size=128)  # 使用输入组成的字典来拟合（必须事先对输入命名）
