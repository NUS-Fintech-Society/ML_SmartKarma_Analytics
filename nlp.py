#!/usr/bin/env python
import re
import string
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import csv
import fasttext
import nltk

from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, BatchNormalization
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import InputSpec, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import io
import torch
import tensorflow as tf
from tqdm import tqdm, trange
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification

data = pd.read_csv("text_labelled.csv")
train, test = sklearn.model_selection.train_test_split(data, random_state = 0)
train.head()

train = train.drop(['Unnamed: 0'], axis=1)
test = test.drop(['Unnamed: 0'], axis=1)
class_count = train["result"].value_counts()
class_count

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

train["text"] = train["text"].apply(preprocess)
test["text"] = test["text"].apply(preprocess)
cv_acc = {}
train.head()

# ## FastText

def train_fasttext(train):
    with open('new_text.csv', mode='w', encoding="utf8", newline='') as new_text:
        for i in range(train.shape[0]):
            employee_writer = csv.writer(new_text, delimiter=',', quotechar='"')
            employee_writer.writerow(train.iloc[i])

    def transform_instance(row):
        cur_row = []
        label = "__label__" + row[1]
        cur_row.append(label)
        cur_row.extend(nltk.word_tokenize(row[0]))
        return cur_row

    def preprocess(input_file, output_file, keep=1):
        with open(output_file, 'w', encoding="utf8") as csvoutfile:
            csv_writer = csv.writer(csvoutfile, delimiter=' ', lineterminator='\n')
            with open(input_file, 'r', newline='', encoding="utf8") as csvinfile:
                csv_reader = csv.reader(csvinfile, delimiter=',', quotechar='"')
                for row in csv_reader:
                    if row[1] in ["0","1"]:
                        row_output = transform_instance(row)
                        csv_writer.writerow(row_output)

    preprocess('new_text.csv', 'trial.train')
    hyper_params = {"lr": 0.01, "epoch": 20, "wordNgrams": 2, "dim": 20}

    model = fastText.train_supervised(input="./trial.train", **hyper_params)
    return model


def validate_fasttext(val, model):
    probs = val["text"].apply(model.predict)
    pred = []
    for i in probs:
        pred.append(int(i[0][0][-1]))

    score = f1_score(val["result"], pred)
    cv_acc["FastText"] = score
    print("CV Accuracy: " + str(score))


def test_fasttext(test, model):
    probs = test["text"].apply(model.predict)
    pred = []
    for i in probs:
        pred.append(int(i[0][0][-1]))
    return pred

model1 = train_fasttext(train)
validate_fasttext(test, model1)


# ## Logistic Regression

def train_logistic(train):
    vectorizer = TfidfVectorizer(use_idf=True, max_df=0.95)
    vectorizer.fit_transform(train["text"].values)
    train_vectorized=vectorizer.transform(train["text"].values)

    logreg = LogisticRegression()
    model = OneVsRestClassifier(logreg)
    model.fit(train_vectorized, train['result'])
    return vectorizer, model

def validate_logistic(val, model, vectorizer):
    val_vectorized = vectorizer.transform(val["text"].values)

    pred = model.predict(val_vectorized)
    score = f1_score(val["result"], pred)
    cv_acc["Logistic Regression"] = score
    print("CV Accuracy: " + str(score))

def test_logistic(test, model, vectorizer):
    test_vectorized = vectorizer.transform(test["text"].values)

    pred = model.predict(test_vectorized)
    return pred

vectorizer, model2 = train_logistic(train)
validate_logistic(test, model2, vectorizer)


# ## Support Vector Machine

def train_svc(train):
    vectorizer = TfidfVectorizer(use_idf=True, max_df=0.95)
    vectorizer.fit_transform(train["text"].values)
    train_vectorized=vectorizer.transform(train["text"].values)

    svc = LinearSVC(dual=False)
    svc.fit(train_vectorized, train['result'])
    return vectorizer, svc

def validate_svc(val, model, vectorizer):
    val_vectorized = vectorizer.transform(val["text"].values)

    pred = model.predict(val_vectorized)
    score = f1_score(val["result"], pred)
    cv_acc["Linear SVM"] = score
    print("CV Accuracy: " + str(score))

vectorizer, model3 = train_svc(train)
validate_svc(test, model3, vectorizer)


# ## Bi-LSTM
def train_BiLSTM(train):
    tk = Tokenizer(lower = True, filters='')
    tk.fit_on_texts(train["text"])
    train_tokenized = tk.texts_to_sequences(train['text'])

    max_len = 100
    X_train = pad_sequences(train_tokenized, maxlen = max_len)

    embed_size = 300
    max_features = 100000
    embedding_path = "./wiki-news-300d-1M-subword.vec"

    def get_coefs(word,*arr):
        return word, np.asarray(arr, dtype='float32')

    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding="utf8"))

    word_index = tk.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    ohe = OneHotEncoder(sparse=False)
    y_ohe = ohe.fit_transform(train['result'].values.reshape(-1, 1))

    def build_model(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32):
        file_path = "best_model.hdf5"
        check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                      save_best_only = True, mode = "min")
        early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

        inp = Input(shape = (max_len,))
        x = Embedding(39457, embed_size, weights = [embedding_matrix], trainable = False)(inp)
        x1 = SpatialDropout1D(spatial_dr)(x)

        x_gru = Bidirectional(LSTM(units, return_sequences = True))(x1)
        x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)
        avg_pool1_gru = GlobalAveragePooling1D()(x1)
        max_pool1_gru = GlobalMaxPooling1D()(x1)

        x_lstm = Bidirectional(LSTM(units, return_sequences = True))(x1)
        x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_lstm)
        avg_pool1_lstm = GlobalAveragePooling1D()(x1)
        max_pool1_lstm = GlobalMaxPooling1D()(x1)

        x = concatenate([avg_pool1_gru, max_pool1_gru,
                        avg_pool1_lstm, max_pool1_lstm])
        x = BatchNormalization()(x)
        x = Dropout(dr)(Dense(dense_units, activation='relu') (x))
        x = Dense(2, activation = "sigmoid")(x)

        model = Model(inputs = inp, outputs = x)
        model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

        history = model.fit(X_train, y_ohe, batch_size = 128, epochs = 5, validation_split=0.1,
                            verbose = 1, callbacks = [check_point, early_stop])
        model = load_model(file_path)
        return model

    model = build_model(lr = 1e-3, lr_d = 1e-10, units = 64, spatial_dr = 0.3,
                      kernel_size1=3, kernel_size2=2, dense_units=32, dr=0.1, conv_size=32)
    return tk, model


# In[8]:


def validate_BiLSTM(val, model, tk):
    max_len = 100
    test_tokenized = tk.texts_to_sequences(val["text"])
    input_test = pad_sequences(test_tokenized, maxlen = max_len)

    probs = model.predict(input_test)
    pred = []
    for i in probs:
        if i[0] > i[1]:
            pred.append(0)
        else:
            pred.append(1)

    score = f1_score(val["result"], pred)
    cv_acc["Bi-LSTM"] = score
    print("CV Accuracy: " + str(score))


# In[9]:


def test_BiLSTM(test, model, tk):
    max_len = 100
    test_tokenized = tk.texts_to_sequences(test["text"])
    input_test = pad_sequences(test_tokenized, maxlen = max_len)

    probs = model.predict(input_test)
    pred = []
    for i in probs:
        if i[0] > i[1]:
            pred.append(0)
        else:
            pred.append(1)
    return pred

tokenizer, model4 = train_BiLSTM(train)
validate_BiLSTM(test, model4, tokenizer)


# ## BERT
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)


def train_BERT(train):
    y_train = train.result.values
    X_train = train.text.values

    X_train = ["[CLS] " + sentence + " [SEP]" for sentence in X_train]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_train = [tokenizer.tokenize(sent) for sent in X_train]

    MAX_LEN = 128
    train_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_train]
    train_ids = pad_sequences(train_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    train_masks = []
    for seq in train_ids:
        seq_mask = [float(i>0) for i in seq]
        train_masks.append(seq_mask)

    train_inputs = torch.tensor(train_ids).to(torch.int64)
    train_labels = torch.tensor(y_train).to(torch.int64)
    train_masks = torch.tensor(train_masks).to(torch.int64)

    batch_size = 32
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)

    t = []
    train_loss_set = []
    epochs = 4

    for _ in trange(epochs, desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
    return model

def validate_BERT(val, model):
    y_val = val.result.values
    X_val = val.text.values

    MAX_LEN = 128
    X_val = ["[CLS] " + sentence + " [SEP]" for sentence in X_val]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_val = [tokenizer.tokenize(sent) for sent in X_val]
    val_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_val]
    val_ids = pad_sequences(val_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    batch_size = 32
    val_masks = []
    for seq in val_ids:
      seq_mask = [float(i>0) for i in seq]
      val_masks.append(seq_mask)

    validation_inputs = torch.tensor(val_ids).to(torch.int64)
    validation_labels = torch.tensor(y_val).to(torch.int64)
    validation_masks = torch.tensor(val_masks).to(torch.int64)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    model.eval()
    predictions , true_labels = [], []

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    score = f1_score(flat_predictions, val["result"])
    cv_acc["BERT"] = score
    print("CV Accuracy: " + str(score))

def test_BERT(test, model):
    X_test = test.text.values
    X_test = ["[CLS] " + sentence + " [SEP]" for sentence in X_test]

    MAX_LEN = 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_test = [tokenizer.tokenize(sent) for sent in X_test]
    test_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_test]
    test_ids = pad_sequences(test_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    batch_size = 32
    test_masks = []
    for seq in test_ids:
        seq_mask = [float(i>0) for i in seq]
        test_masks.append(seq_mask)

    test_inputs = torch.tensor(test_ids).to(torch.int64)
    test_masks = torch.tensor(test_masks).to(torch.int64)

    test_data = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    model.eval()
    predictions , true_labels = [], []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    return flat_predictions

model5 = train_BERT(train)

validate_BERT(test, model5)

cv_acc = {k: v for k, v in sorted(cv_acc.items(), key=lambda item: item[1], reverse=True)}
data = pd.DataFrame({'Model':list(cv_acc.keys()), 'CV Scores':list(cv_acc.values())})

