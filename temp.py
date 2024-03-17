import pickle
from typing import IO

import pandas as pd
from pandas import DataFrame
from transformers import TFBertTokenizer, TFBertModel
import gc

def read_dataset_fake(name: str) -> DataFrame:
    df = pd.read_csv(f"{name}")
    cats = ['News', 'politics', 'Government News', 'left-news', 'US_News', 'Middle-east']
    cond = ~df['subject'].isin(cats)
    df.loc[(cond), 'text'] += ' ' + df.loc[(cond), 'subject']
    df.loc[(cond), 'subject'] = df.loc[(cond), 'date']
    date1 = pd.to_datetime(df['date'], format='%B %d, %Y', errors='coerce')
    date2 = pd.to_datetime(df['date'], format='%b %d, %Y', errors='coerce')
    date3 = pd.to_datetime(df['date'], format='%d-%b-%y', errors='coerce')
    date4 = date1.fillna(date2).fillna(date3)
    date1 = pd.to_datetime(df['Unnamed: 4'], format='%B %d, %Y', errors='coerce')
    date2 = pd.to_datetime(df['Unnamed: 4'], format='%b %d, %Y', errors='coerce')
    date5 = date1.fillna(date2)
    date4 = date4.fillna(date5)
    df['date'] = date4

    return df[['title', 'text', 'subject', 'date']]


def read_dataset_real(name: str) -> DataFrame:
    df = pd.read_csv(f"{name}")
    date1 = pd.to_datetime(df['date'], format='%B %d, %Y ')
    df['date'] = date1

    return df


def get_dataset() -> DataFrame:
    fake = read_dataset_fake('dataset/fake.csv')#.tail(15)
    real = read_dataset_real('dataset/true.csv')#.tail(15)
    fake["class"] = 0
    real["class"] = 1
    df = pd.concat([real, fake])
    # Removing last 10 rows for manual testing
    df_fake_manual_testing = fake.tail(10)
    for i in range(23480, 23470, -1):
        fake.drop([i], axis=0, inplace=True)

    df_true_manual_testing = real.tail(10)
    for i in range(21416, 21406, -1):
        real.drop([i], axis=0, inplace=True)

    df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
    df_manual_testing.to_csv("manual_testing.csv")

    # shuffle
    df = df.sample(frac=1)
    df.reset_index(inplace=True)
    y = df['class'].tolist()
    write_list(y, 'y.plk')
    df.drop(["index", "class"], axis=1, inplace=True)
    return df


def normalize_dates(df: DataFrame):
    oldest = min(df)
    newest = max(df)
    df = [(i - oldest).days / (newest - oldest).days for i in df]
    write_list(df, 'days.plk')


def preprocess_data(df: DataFrame):  # -> Tuple[List, List, List]:
    # one hot encoding categorical
    df = pd.get_dummies(df, columns=['subject'])
    # normalize dates
    normalize_dates(df['date'])
    model_name = 'bert-base-uncased'
    # For more details - https://huggingface.co/bert-base-uncased
    model = TFBertModel.from_pretrained(model_name)
    tokenizer = TFBertTokenizer.from_pretrained(model_name)

    # text =tf.Variable( "Replace me by any text you'd like.")
    # encoded_input = tokenizer(text)
    # output = model(encoded_input)

    # tokenize and get embeddings
    def write_embedding_in_file(f: IO, x: DataFrame):
        embd = model(input_ids=x['input_ids'], attention_mask=x['attention_mask'],
                     token_type_ids=x['token_type_ids']).pooler_output[0].numpy().tolist()
        pickle.dump(embd, f)
        f.flush()
        return 1

    f = 'titles_embeddings_temp.plk'
    # with open(f, 'ab') as fp:
    #     df['title'].apply(lambda x: write_embedding_in_file(fp, tokenizer([x], truncation=True, padding="max_length")))

    f = 'text_embeddings_temp.plk'
    with (open(f, 'ab') as fp):
        df['text'].apply(lambda x: write_embedding_in_file(fp, tokenizer(get_split(x), truncation=True, padding="max_length")))

    # with open(f, 'wb') as fp:
    #     titles = df['title'].map(lambda x: tokenizer([x], truncation=True, padding="max_length")).map(lambda x: model(
    #         input_ids=x['input_ids'], attention_mask=x['attention_mask'],
    #         token_type_ids=x['token_type_ids']).last_hidden_state[0].numpy().tolist())
    # for tit in df[['title', 'text']].tolist():
    #     titles.append(tokenizer(tit, truncation=True, padding="max_length"))
    # text.append(tokenizer(tex, truncation=True, padding="max_length"))
    # get embeddings
    # input_ids = []
    # att_mask = []
    # token_type_ids = []
    # for x in titles:
    #     input_ids.append(x['input_ids'])
    #     att_mask.append(x['attention_mask'])
    #     token_type_ids.append(x['token_type_ids'])
    # att_mask = [x['attention_mask'] for x in titles]
    #
    # outputs = model(input_ids=tf.stack(titles['input_ids'], 1)[0],
    #                 attention_mask=tf.stack(titles['attention_mask'], 1)[0],
    #                 token_type_ids=tf.stack(titles['token_type_ids'], 1)[0])
    # titles = outputs.last_hidden_state[0]

    # input_ids = tokenizer.convert_tokens_to_ids(text)
    # outputs = model(input_ids=tf.stack(text['input_ids'], 1)[0], attention_mask=tf.stack(text['attention_mask'], 1)[0],
    #                 token_type_ids=tf.stack(text['token_type_ids'], 1)[0])
    # text = outputs.last_hidden_state[0]

    # x = tf.stack((titles, text), axis=0)
    filenames = ['titles_embeddings.plk', 'text_embeddings.plk', 'days.plk', 'y.plk']


def get_split(text1, n_words=510):
    l_total = []
    splited = text1.split()

    n = len(splited) // n_words

    for w in range(n+1):
        l_partial = splited[w * n_words:(w + 1) * n_words]
        l_total.append(" ".join(l_partial))
    return l_total


# write list to binary file
def write_list(a_list, filename):
    # store list in binary file so 'wb' mode
    with open(filename, 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')


def read_pkl(filename):
    # for reading also binary mode is important
    res = []
    with open(filename, "rb") as fp:
        while True:
            try:
                res.append(pickle.load(fp))  # Need to call pickle.load() for each pickle.dump() used to make the file
            except EOFError:
                break
    return res


def read_data_from_pickle():
    filenames = ['titles_embeddings.plk', 'text_embeddings.plk', 'days.plk', 'y.plk']
    t = tuple()
    for f in filenames:
        t = (*t, read_pkl(f))
    return t

if __name__ == '__main__':

    new_run = True
    if new_run:
        x = get_dataset()
        preprocess_data(x)
    lst = read_pkl('/home/milad/Desktop/titles_embeddings.plk')
    del lst

    print('somethign')

    gc.collect()
    print('so')