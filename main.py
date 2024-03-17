from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Concatenate, Input, Dropout
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split

from dataset_preprocessing import *

MODEL_NAME = 'model.keras'


def create_model_and_train(new_run=False):
    if new_run:
        x = get_dataset()
        preprocess_data(x)
    print('load files')
    titles, text, days, y = read_data_from_pickle()
    days = days[0]
    y = y[0]
    print('split dataset')
    titles_train, titles_test, text_train, text_test, y_train, y_test = train_test_split(titles, text, y,
                                                                                         test_size=0.25)
    print('prepare model input')
    titles_train = tf.convert_to_tensor(titles_train)
    text_train = tf.ragged.constant(text_train).to_tensor()
    y_train = tf.convert_to_tensor(y_train)
    print('create model')
    model = prepare_model()
    print('start taining')
    train(model, [titles_train, text_train], y_train)

    print('somethign')


def prepare_for_inference() -> List:
    df = pd.read_csv('manual_testing.csv')
    df = pd.get_dummies(df, columns=['subject'])
    # normalize dates
    # normalize_dates(df['date'])
    model_name = 'bert-base-uncased'
    # For more details - https://huggingface.co/bert-base-uncased
    model = TFBertModel.from_pretrained(model_name)
    tokenizer = TFBertTokenizer.from_pretrained(model_name)

    # text =tf.Variable( "Replace me by any text you'd like.")
    # encoded_input = tokenizer(text)
    # output = model(encoded_input)

    # tokenize and get embeddings
    def write_embedding_in_file(x: DataFrame):
        embd = model(input_ids=x['input_ids'], attention_mask=x['attention_mask'],
                     token_type_ids=x['token_type_ids']).pooler_output.numpy().tolist()

        return embd

    # f = 'titles_embeddings.plk'
    # with open(f, 'ab') as fp:
    titles_embeddings = df['title'].apply(
            lambda x: write_embedding_in_file(tokenizer([x], truncation=True, padding="max_length"))).to_list()

    text_embeddings = df['text'].apply(
            lambda x: write_embedding_in_file(tokenizer(get_split(x), truncation=True, padding="max_length"))).to_list()


    y = df['class'].tolist()
    titles_embeddings = tf.squeeze(tf.convert_to_tensor(titles_embeddings))
    text_embeddings = tf.ragged.constant(text_embeddings).to_tensor(shape=[len(text_embeddings), 16, 768])
    y = tf.convert_to_tensor(y)
    model = get_model()
    res = model([titles_embeddings, text_embeddings])
    print('sososo')
    return res

def prepare_for_train() -> DataFrame:
    pass


def prepare_model():
    titles = Input(shape=(768), dtype=tf.float32)
    text = Input(shape=[16, 768], dtype=tf.float32, ragged=False)
    x = Conv1D(256, 4, activation='relu')(text)
    x = MaxPooling1D(5)(x)
    x = Flatten()(x)
    y = Dense(128, activation='relu')(titles)

    x = Concatenate(-1)([y, x])
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[titles, text], outputs=x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'crossentropy'])
    model.summary()
    return model


def train(model: Model, x: List, y: tf.Tensor):
    history = model.fit(x=x, y=y, epochs=20, batch_size=16)
    model.save(MODEL_NAME)
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return model


def get_model():
    return load_model(MODEL_NAME)


if __name__ == '__main__':
    # create_model_and_train()
    res = prepare_for_inference()
    print('end')