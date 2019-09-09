# -*- coding: UTF-8 -*-

from gensim.models import word2vec
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

from keras.layers import Dense, Input
from keras.models import Model
from keras.layers import Embedding, Conv2D, MaxPooling2D, Dropout, concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import regularizers

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.utils import plot_model

import matplotlib.pyplot as plt


def main():
    news = pd.read_csv('data/data_seged_monpa.csv')
    news_tag = news[['text', 'replyType', 'seg_text']]
    news_tag = news_tag[news_tag['replyType'] != 'NOT_ARTICLE']
    types = news_tag.replyType.unique()
    dic = {}
    for i, types in enumerate(types):
        dic[types] = i
    print(dic)
    news_tag['type_id'] = news_tag.replyType.apply(lambda x: dic[x])
    labels = news_tag.replyType.apply(lambda x: dic[x])
    news_tag = find_null(news_tag)
    X = news_tag.seg_text
    y = news_tag.type_id
    print(y.value_counts())
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    print(X_train.shape, 'training data ')
    print(X_test.shape, 'testing data')
    X_train = transfer_lsit(X_train)
    X_test = transfer_lsit(X_test)
    all_data = pd.concat([X_train, X_test])

    # embedding setting
    EMBEDDING_DIM = 100
    NUM_WORDS = 2764036
    vocabulary_size = NUM_WORDS
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    word_vectors = word2vec.Word2Vec.load("output/word2vec.model")
    embedding_matrix = to_embedding(EMBEDDING_DIM, NUM_WORDS, vocabulary_size,
                                    embedding_matrix, word_vectors, X_train,
                                    X_test)
    del (word_vectors)

    embedding_layer = Embedding(vocabulary_size,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=True)

    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'')
    tokenizer.fit_on_texts(all_data.values)

    train_text = X_train.values
    train_index = X_train.index

    sequences_train = tokenizer.texts_to_sequences(train_text)

    X_train = pad_sequences(sequences_train, maxlen=600)

    y_train = to_categorical(np.asarray(labels[train_index]))

    print('Shape of X train:', X_train.shape)
    print('Shape of label train:', y_train.shape)

    test_text = X_test.values
    test_index = X_test.index
    sequences_test = tokenizer.texts_to_sequences(test_text)
    X_test = pad_sequences(sequences_test, maxlen=X_train.shape[1])
    y_test = to_categorical(np.asarray(labels[test_index]))

    sequence_length = X_train.shape[1]
    filter_sizes = [2, 3, 4]
    num_filters = 128
    drop = 0.2
    penalty = 0.0001

    inputs = Input(shape=(sequence_length, ))
    embedding = embedding_layer(inputs)
    reshape = Reshape((sequence_length, EMBEDDING_DIM, 1))(embedding)

    conv_0 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),
                    activation='softmax',
                    kernel_regularizer=regularizers.l2(penalty))(reshape)
    conv_1 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),
                    activation='relu',
                    kernel_regularizer=regularizers.l2(penalty))(reshape)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),
                    activation='relu',
                    kernel_regularizer=regularizers.l2(penalty))(reshape)

    maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1),
                             strides=(1, 1))(conv_0)

    maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1),
                             strides=(1, 1))(conv_1)
    maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1),
                             strides=(1, 1))(conv_2)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
    dropout = Dropout(drop)(merged_tensor)
    flatten = Flatten()(dropout)
    reshape = Reshape((3 * num_filters, ))(flatten)
    output = Dense(units=2,
                   activation='softmax',
                   kernel_regularizer=regularizers.l2(penalty))(reshape)

    # this creates a model that includes
    model = Model(inputs, output)
    model.summary()

    adam = Adam(lr=1e-3)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
    callbacks = [EarlyStopping(monitor='val_loss')]
    history = model.fit(X_train,
                        y_train,
                        batch_size=64,
                        epochs=50,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=callbacks)

    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
    print(matrix)

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig("output/acc.png")
    score, acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', acc)

    plot_model(model,
               to_file='output/model.png',
               show_shapes=False,
               show_layer_names=False)


def to_embedding(EMBEDDING_DIM, NUM_WORDS, vocabulary_size, embedding_matrix,
                 word_vectors, X_train, X_test):
    i = 0

    for sentence in X_train:
        for word in sentence:
            try:
                embedding_vector = word_vectors[word]
                embedding_matrix[i] = embedding_vector
                i = i + 1
            except KeyError:
                embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25),
                                                       EMBEDDING_DIM)
                i = i + 1

    for sentence in X_test:
        for word in sentence:
            try:
                embedding_vector = word_vectors[word]
                embedding_matrix[i] = embedding_vector
                i = i + 1
            except KeyError:
                embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25),
                                                       EMBEDDING_DIM)
                i = i + 1

    return embedding_matrix


def transfer_lsit(filename):
    seq = filename
    for ids in seq.index:
        seq[ids] = eval(seq[ids])
    return seq


def find_length(X_train, X_test):
    leng = []
    for index in X_train:
        leng.append(len(index))
    for index in X_test:
        leng.append(len(index))
    pand = pd.Series(leng)
    mean_v = pand.mean()
    return int(mean_v)


def find_null(news_tag):
    word_count = []
    null_number = 0
    for index in news_tag.seg_text:
        text_len = len(eval(index))
        if text_len == 0:
            null_number += 1
        word_count.append(text_len)
    news_tag['word_count'] = word_count
    news_tag = news_tag[news_tag.word_count > 0]
    print('remove %d null items' % null_number)
    return news_tag


if __name__ == "__main__":
    main()
