
# coding: utf-8

# # Реккурентные нейронные сети для извлечения именованных сущностей

# In[62]:

from keras.preprocessing import sequence
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, concatenate
from keras.layers import *
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import *


# ## Данные

# In[2]:

df = pd.read_csv('../data/ner.csv', encoding = "ISO-8859-1", error_bad_lines=False)

df.head()


# In[3]:

data = []
sent = []

for index, item in df.iterrows():    
    if (item.word in ['.', '?', '!', '...']):
        sent.append([str(item.lemma),str(item.tag)])
        data.append(sent)
        sent = []
    else:
        sent.append([str(item.word),str(item.tag)])


# In[4]:

data[0]


# Предобработка: формируем признаки и целевую переменную, находим уникальное количество слов и максимальную длину предложения:

# In[5]:

lengths = [len(x) for x in data]
X = [' '.join([word[0] for word in sent]) for sent in data]
y = [[word[1] for word in sent] for sent in data]

all_text = [word for sent in X for word in sent]
words = list(set(all_text))
maxlen = max([len(x.split()) for x in X])


# Последовательности для обучения:

# In[6]:

tokenizer = Tokenizer(num_words=len(words))
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(sequences, maxlen=maxlen)


# Преобразования целевого признака: посчитаем сколько раз встречается каждый тег

# In[7]:

from nltk import FreqDist
fd = FreqDist()
for i in y:
    fd.update(i)

n_tags = len(list(fd.keys()))


# Подготавливаем последовательности:

# In[8]:

encoder = LabelEncoder()
encoder.fit(list(fd.keys()))
y_enc = [encoder.transform(sent) for sent in y] 
y_padded = pad_sequences(y_enc, maxlen=maxlen, value = list(encoder.classes_).index('O'))
y_cat  = [to_categorical(sent, num_classes=n_tags) for sent in y_padded]


# In[9]:

n_tags  = len(fd.keys())


# In[10]:

print(y[0])
print(y_enc[0])
print(y_padded[0])
print(y_cat[0])
print(len(y_enc[0]))
print(len(y_cat[0][0]))
print(len(list(fd.keys())))


# Размерности данных:

# In[11]:

(X_train, X_test, Y_train, Y_test) = train_test_split(X_padded, np.asarray(y_cat), test_size=0.332, random_state=42)

print ('Train data shapes:')
print(len(X_train), len(X_train[0]), len(Y_train), len(Y_train[0]))
print ('Test data shapes:')
print(len(X_test), len(X_test[0]), len(Y_test), len(Y_test[0]))


# Параметры обучения:

# In[12]:

max_features = len(words)
embedding_size = 100
hidden_size = 32
out_size = n_tags
nb_epoch = 20
batch_size = 32


# Нейронная сеть:

# In[ ]:

# model = Sequential()
# model.add(Embedding(max_features, embedding_size, input_length=maxlen))
# model.add(Bidirectional(LSTM(hidden_size, return_sequences = True)))
# model.add(TimeDistributed(Dense(out_size, activation = 'softmax')))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# for i in range(nb_epoch):
#     model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)
#     model.save('../models/bilstm_'+str(i)+'.h5')
# model = load_model('../models/bilstm_'+str(19)+'.h5')


# # In[14]:

# score = model.evaluate(X_test, Y_test)

# print('Test score:', score[0])
# print('Test accuracy:', score[1])


# # In[15]:

# pred = model.predict(X_test)


# # In[16]:

# pred_argmax = [[np.argmax(word) for word in sent] for sent in pred]
# y_pred = encoder.inverse_transform(pred_argmax)
# Y_test_argmax = [[np.argmax(word) for word in sent] for sent in Y_test]
# y_true = encoder.inverse_transform(Y_test_argmax)


# # In[17]:

# from sklearn_crfsuite import metrics

# labels=list(fd.keys())

# sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))

# print(metrics.flat_classification_report(y_true, y_pred, labels = sorted_labels , digits=3))


# # In[18]:

# y_true_flat = metrics.flatten(y_true)
# y_pred_flat = metrics.flatten(y_pred)


# # In[19]:

# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import *
# get_ipython().magic('matplotlib inline')


# sns.heatmap(data=confusion_matrix(y_true_flat, y_pred_flat), annot=True, fmt="d", cbar=False, xticklabels=list(encoder.classes_), yticklabels=list(encoder.classes_))
# plt.title("Confusion matrix")
# plt.show()


# # ## Задание
# # 
# # Добавьте POS-тэги к обучению: 
# # * посчитайте, сколько всего тегов использовано для разметки текстов;
# # * пронумеруйте теги и замените каждый тег на его порядковый номер;
# # * создайте новый вход для нейронной сети с эмбеддингами тегов;
# # * конкатинируйте эмбеддинги слов и эмбеддинги тегов, используя keras.layers.Concatenate(axis=-1)
# # * используйте конкатерированные эмбеддинги в качестве входа сети 

# # In[20]:

# data = []
# sent = []

# for index, item in df.iterrows():    
#     if (item.word in ['.', '?', '!', '...']):
#         sent.append(item.pos)
#         data.append(sent)
#         sent = []
#     else:
#         sent.append(item.pos)

# data[0]


# # In[36]:

# from nltk import FreqDist
# fd = FreqDist()
# for sent in data:
#     fd.update(sent)
# n_pos_tags = len(list(fd.keys()))
# pos_tags = list(fd.keys())


# # In[30]:

# len(data)


# # In[38]:

# encoder = LabelEncoder()
# pos_tags.append('O')
# encoder.fit(pos_tags)
# pos_enc = [encoder.transform(sent) for sent in data] 
# pos_padded = pad_sequences(pos_enc, maxlen=maxlen, value = list(encoder.classes_).index('O'))
# pos_cat  = [to_categorical(sent, num_classes=n_pos_tags+1) for sent in pos_padded]


# # In[42]:

# pos_train, post_test = train_test_split(pos_cat, test_size=0.332, random_state=42)


# # In[54]:

# word_model = Sequential()
# word_model.add(Embedding(max_features, embedding_size, input_length=maxlen))


# pos_model = Sequential()
# pos_model.add(Embedding(max_features, 40, input_length=maxlen))


# # In[ ]:

# model = Model(inputs = Concatenate([word_model, pos_model]))
# model.add()
# model.add(Bidirectional(LSTM(hidden_size, return_sequences = True)))
# model.add(TimeDistributed(Dense(out_size, activation = 'softmax')))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# for i in range(nb_epoch):
#     model.fit([X_train, pos_train], Y_train, batch_size=batch_size, epochs=1)


# ## Модель biLSTM-CRF

# In[ ]:

from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Bidirectional(LSTM(hidden_size, return_sequences = True)))
model.add(TimeDistributed(Dense(hidden_size, activation = 'softmax')))

crf = (CRF(n_tags))
model.add(crf)

model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

for i in range(nb_epoch):
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)
    save_load_utils.save_all_weights(model,'../models/bilstm_crf_'+str(i)+'.h5')






# In[ ]:

score = model.evaluate(X_test, Y_test)

print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[ ]:

pred = model.predict(X_test)


# In[ ]:

pred_argmax = [[np.argmax(word) for word in sent] for sent in pred]
y_pred = encoder.inverse_transform(pred_argmax)
Y_test_argmax = [[np.argmax(word) for word in sent] for sent in Y_test]
y_true = encoder.inverse_transform(Y_test_argmax)


# In[ ]:

print(metrics.flat_classification_report(y_true, y_pred, labels=list(fd.keys()), digits=3))


# In[ ]:

sns.heatmap(data=confusion_matrix(y_true_flat, y_pred_flat), annot=True, fmt="d", cbar=False, xticklabels=list(encoder.classes_), yticklabels=list(encoder.classes_))
plt.title("Confusion matrix")
plt.show()

