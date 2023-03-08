import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import math
import seaborn as sn
import tensorflow as tf
from bs4 import BeautifulSoup
from sklearn import metrics as sk_metrics
import nltk
import re

df = pd.read_csv('Posts.xml', encoding='utf-8')
df['Y_cat'] = df['Y'].astype('category').cat.codes
df['CreationDateTime'] = pd.to_datetime(df['CreationDate']).dt.hour
df['title_body'] = df['Title'] + ' ' + df['Body']
df.sample(10)

#Data Preprocessing
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

def data_cleaning(data):
    data = data.lower()
    data = re.sub(r'([^a-zA-Z\s])', '', data)
    data = data.split()
    temp = []
    for i in data:
        if i not in stop_words:
            temp.append(i)
    data = ' '.join(temp)
    return data

y = np.array(df['Y_cat'], dtype=np.int64)
x = (str(df['CreationDateTime']) + ' ' + df['title_body'].apply(lambda x: data_cleaning(x))).values

train_size = int(df.shape[0] * 0.8)
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

#Build a LSTM model from scratch
# keras tokenizer
MAX_WORDS = 20000

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=MAX_WORDS, 
    oov_token=MAX_WORDS+1,
    
    )
# fit on training data
tokenizer.fit_on_texts(x_train)

# convert text to tokenized sequences
x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)

# add padding
x_train_padded = tf.keras.preprocessing.sequence.pad_sequences(x_train_tokens, maxlen=200, padding='post')
x_test_padded = tf.keras.preprocessing.sequence.pad_sequences(x_test_tokens, maxlen=200, padding='post')

# initiate tf datasets
ds_train = tf.data.Dataset.from_tensor_slices((x_train_padded, y_train)).batch(64)
ds_test = tf.data.Dataset.from_tensor_slices((x_test_padded, y_test)).batch(64)

# model parameters
embedding_dim = 512
vocab_size = MAX_WORDS + 2
n_classes = df['Y_cat'].nunique()

# build the model
inputs = tf.keras.Input(shape=(None,), dtype="int32")
x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

# compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# call bacls
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=2, 
    min_lr=0.0001)

# fit the model
results = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=100,
    callbacks=[early_stop_callback, reduce_lr]
    )

from tokenizers import BertWordPieceTokenizer
import transformers    
# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased' , lower=True)
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=True)
fast_tokenizer

def fast_encode(texts, tokenizer, chunk_size=256, maxlen=200):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(length=maxlen)
    all_ids = []
    
    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)

max_len=200

x_train = fast_encode(x_train, fast_tokenizer, maxlen=max_len)
x_test = fast_encode(x_test, fast_tokenizer, maxlen=max_len)
bert_transformer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32)
sequence_output = bert_transformer(input_word_ids)[0]
cls_token = sequence_output[:, 0, :]
out = tf.keras.layers.Dense(3, activation='softmax')(cls_token)

model = tf.keras.Model(inputs=input_word_ids, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=7e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(
    x_train,
    y_train,
    batch_size=16,
    validation_data=(x_test,y_test),
    epochs=5)