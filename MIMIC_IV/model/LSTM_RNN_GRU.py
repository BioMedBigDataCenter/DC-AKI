#!/usr/bin/env python
# coding: utf-8

import os
import gc
import pandas as pd
import numpy as np
from datetime import datetime
from math import ceil
from tqdm import tqdm
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, accuracy_score, auc, f1_score, matthews_corrcoef, precision_score, precision_recall_curve, confusion_matrix

import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Input, GRU
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from tensorflow.compat.v1.keras.layers import LSTM

seed = 8888
os.environ['PYTHONHASHSEED']=str(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

to_save = {}
to_save['random_seed'] = seed

def find_optimal_cutoff(TPRs, FPRs, thresholds):
    """
    Find the optimal classification threshold, where the distance between TPR and FPR is maximum
    :param TPRs: TPR values at different thresholds
    :param FPRs: FPR values at different thresholds
    :param thresholds: Different classification thresholds
    :return: optimal classification threshold, (optimal TPR, optimal FPR)
    """
    distance = TPRs - FPRs
    idx = np.argmax(distance)  # Only the first occurrence is returned.
    optimal_threshold = thresholds[idx]
    optimal_point = (TPRs[idx], FPRs[idx])
    
    return optimal_threshold, optimal_point


def roc_values(y_true, y_proba):
    """
    Get FPRs, TPRs, AUC value, optimal classification threshold, optimal TPR and optimal FPR of ROC curve
    :param y_true: True values list of sample labels
    :param y_proba: Model-predicted probability of AKI
    :return: FPRs, TPRs, AUC value, optimal classification threshold, (optimal TPR, optimal FPR)
    """
    fprs, tprs, thresholds = roc_curve(y_true, y_proba, pos_label=1)
    roc_auc = auc(fprs, tprs)
    optimal_threshold, optimal_point = find_optimal_cutoff(TPRs=tprs, FPRs=fprs, thresholds=thresholds)
    
    return fprs, tprs, roc_auc, optimal_threshold, optimal_point


def get_set(id_list, id_col, time_col, data):
    """
    Get patient data for the specified ID list
    :param id_col: Name of ID column
    :param time_col: Name of timestamp column
    :param data: DataFrame of patient data
    :return: Patient data for the specified ID list
    """
    data = data[data[id_col].isin(id_list)].sort_values(by=[id_col, time_col], ascending=True)
    data.reset_index(drop=True, inplace=True)
    
    return data


def performance(label, proba, pred):
    """
    Get model evaluation metrics
    :param label: True values list of sample labels
    :param proba: Model-predicted probability of AKI
    :param pred: Predicted values under the optimal classification threshold
    :return: Accuracy, Recall, MCC, Precision, F1-score, AUC, FPR
    """
    acc = accuracy_score(label, pred)
    recall = recall_score(label, pred, pos_label=1)
    mcc = matthews_corrcoef(label, pred)
    precision = precision_score(label, pred)
    f1 = f1_score(label, pred)
    roc_auc = roc_auc_score(label, proba)
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    fpr = fp/(fp+tn)
    
    print(f'accuracy: {acc:.4f}')
    print(f'recall: {recall:.4f}')
    print(f'AUC: {roc_auc:.4f}')
    print(f'MCC: {mcc:.4f}')
    print(f'precision: {precision:.4f}')
    print(f'F1: {f1:.4f}')
    print(f'FPR: {fpr:.4f}')
    
    return acc, recall, mcc, precision, f1, roc_auc, fpr


BATCH_SIZE = 128
EPOCHS = 100

# Imputing(feature and label)
FILL_WITH_ZERO = True
FILL_VALUE = 0

LABEL_COLUMN = 'aki_stage'
# Resample interval
RSMP = 6
# Advance prediction time
SFT = 48
# FEATURE_SET = 'SCr_lab_vit_dem_med_pro'

DATA_PATH = f'./data_con.tsv'
#DATA_PATH = f'./con0726/data_con_eICU0814.tsv'
DATA_EXTERNAL_PATH = f'./data0727/data_external_rsmp{RSMP}.tsv'
MODEL_SAVE_PATH = './GRU24.h5'
TEST_RESULT_PATH = f'./predict_result/rsmp{RSMP}_sft{SFT}_continuous.csv'
EXTERNAL_RESULT_PATH = f'./predict_result/rsmp{RSMP}_sft{SFT}_continuous_external.csv' 
TRAINING_PARAM_PATH = './result.xls'

to_save['batch_size'] = BATCH_SIZE
to_save['epoch'] = EPOCHS
to_save['imputation'] = 'fill with zero'
to_save['resample_interval'] = RSMP
to_save['prediction_ahead'] = SFT
to_save['data_dev_path'] = DATA_PATH
to_save['data_external_path'] = DATA_EXTERNAL_PATH
to_save['model_path'] = MODEL_SAVE_PATH
to_save['test_result_path'] = TEST_RESULT_PATH
to_save['external_result_path'] = EXTERNAL_RESULT_PATH

data_ori = pd.read_csv(DATA_PATH, sep='\t')
if 'patient_id' in data_ori.columns:
    data_ori.rename(columns={'patient_id': 'stay_id', '检验日期': 'charttime'}, inplace=True)
id_col, time_col = 'stay_id', 'charttime'
print(f'Sample num: {data_ori.stay_id.nunique()}')
data_ori.shape

data_ori.head()

data_ori.columns

data_ori = data_ori.rename(columns={'akistage': 'aki_stage'})
others = [id_col, time_col, LABEL_COLUMN]
# common_feat = ['hemoglobin', 'chloride', 'sodium', 'potassium', 'creatinine', 'admission_age', 'gender_F', 'gender_M']
# data_ori = data_ori[['icustay_id', 'charttime', 'aki_stage']+common_feat].copy()


data = data_ori.copy()
# Label imputing
data[LABEL_COLUMN].fillna(FILL_VALUE, inplace=True)
data[LABEL_COLUMN] = data.groupby(id_col)[LABEL_COLUMN].shift(-(SFT/RSMP))
data.dropna(subset=[LABEL_COLUMN], how='all', inplace=True)
data[LABEL_COLUMN].replace([2, 3], [1, 1], inplace=True)
    
print(f'Sample_num: {data[id_col].nunique()}')
print(f'AKI_label: {data[LABEL_COLUMN].unique()}')
data.shape

icustays = data.stay_id.unique().tolist()
features = data.columns.difference(others).tolist()

print(f'Sample num: {len(icustays)}, Feature num: {len(features)}\n')
print(f'features: {features}')

to_save['sample_num'] = len(icustays)
to_save['feature_num'] = len(features)

new_column_order = [col for col in data.columns if col != 'aki_stage'] + ['aki_stage']
data = data[new_column_order]

column_order = data.columns.tolist()
print(column_order)

data.head()

# # Imputing
print('Imputing with zero!')
data.fillna(FILL_VALUE, inplace=True)


# # Train/Validation/Test set
train_id, val_test_id = train_test_split(icustays, test_size=0.2, random_state=seed)
val_id, test_id = train_test_split(val_test_id, test_size=0.5, random_state=seed)
print(f'Train samples: {len(train_id)}\nValidation samples: {len(val_id)}\nTest samples: {len(test_id)}')
train = get_set(id_list=train_id, id_col=id_col, time_col=time_col, data=data)
val = get_set(val_id, id_col, time_col, data)
test = get_set(test_id, id_col, time_col, data)

train.shape, val.shape, test.shape

# 各集合中AKI病人占比
aki_train = train[[id_col, LABEL_COLUMN]].copy().groupby([id_col]).max()
aki_val = val[[id_col, LABEL_COLUMN]].copy().groupby([id_col]).max()
aki_test = test[[id_col, LABEL_COLUMN]].copy().groupby([id_col]).max()

print(f'train: {aki_train.aki_stage.sum()/len(aki_train):.4f}')
print(f'val: {aki_val.aki_stage.sum()/len(aki_val):.4f}')      
print(f'test: {aki_test.aki_stage.sum()/len(aki_test):.4f}')


# # Split features and label / Normalization
train.sort_values(by=[id_col, time_col], ascending=True, inplace=True)
val.sort_values(by=[id_col, time_col], ascending=True, inplace=True)
test.sort_values(by=[id_col, time_col], ascending=True, inplace=True)
train = train[others+features].copy()
val = val[others+features].copy()
test = test[others+features].copy()
train_list = [train[train[id_col]==i].iloc[:, 2:].values for i in train_id]
val_list = [val[val[id_col]==i].iloc[:, 2:].values for i in val_id]
test_list = [test[test[id_col]==i].iloc[:, 2:].values for i in test_id]

# 将训练集中最长的时间序列长度设置为模型最长时间步
maxlen = train.groupby('stay_id').size().max()
to_save['time_seq_length'] = maxlen
maxlen
train_data = pad_sequences(
    train_list, 
    dtype='float32', 
    padding='pre', 
    truncating='pre', 
    value=0, 
    maxlen=maxlen
)
val_data = pad_sequences(
    val_list, 
    dtype='float32', 
    padding='pre', 
    truncating='pre', 
    value=0, 
    maxlen=maxlen
)
test_data = pad_sequences(
    test_list, 
    dtype='float32', 
    padding='pre', 
    truncating='pre', 
    value=0, 
    maxlen=maxlen
)
train_data.shape, val_data.shape, test_data.shape

# 将特征和标签分离
train_label = train_data[:, :, 0]
val_label = val_data[:, :, 0]
test_label = test_data[:, :, 0]

train_data = train_data[:, :, 1:]
val_data = val_data[:, :, 1:]
test_data = test_data[:, :, 1:]

# feature normalization
train_data = train_data.reshape(-1, len(features))
val_data = val_data.reshape(-1, len(features))
test_data = test_data.reshape(-1, len(features))

feature_scaler = StandardScaler()
feature_scaler.fit(train_data)

train_data = feature_scaler.transform(train_data)
train_data = train_data.reshape(-1, maxlen, len(features))

val_data = feature_scaler.transform(val_data)
val_data = val_data.reshape(-1, maxlen, len(features))

test_data = feature_scaler.transform(test_data)
test_data = test_data.reshape(-1, maxlen, len(features))

train_data.shape, val_data.shape, test_data.shape

# label one-hot
train_label = train_label.reshape(-1, maxlen, 1)
val_label = val_label.reshape(-1, maxlen, 1)
test_label = test_label.reshape(-1, maxlen, 1)

train_label = to_categorical(train_label)
val_label = to_categorical(val_label)
test_label = to_categorical(test_label)

train_label.shape, val_label.shape, test_label.shape


# # LSTM
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GRU, Bidirectional, Dense, Dropout, Attention
def get_model(max_len, feature_num):
    
    clear_session()
    inputs = Input(shape=(max_len, feature_num), dtype='float32')
    x = Dense(64, activation='relu')(inputs)
    x = LSTM(32, return_sequences=True,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = LSTM(32, return_sequences=True,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = LSTM(32, return_sequences=True,activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(2, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.summary()
    
    return model
model_LSTM = get_model(max_len=maxlen, feature_num=len(features))
adam = optimizers.Adam(lr=0.001)
loss = BinaryCrossentropy(from_logits=False)
k_auc = AUC()

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH, 
    monitor='val_auc', 
    mode='max', 
    save_best_only=True
)

model_LSTM.compile(
    optimizer=adam, 
    loss=loss,
    metrics=[k_auc]
)

to_save['lr'] = 0.001
time_start = datetime.now()
history = model_LSTM.fit(
    train_data, train_label,
    validation_data=(val_data, val_label), 
    batch_size=2048, 
    epochs=1, 
    callbacks=[checkpoint])
train_time = (datetime.now() - time_start).total_seconds()
to_save['train_time'] = train_time
print(train_time)
test_LSTM = model_LSTM.predict(test_data)
test_LSTM.shape,test_label.shape
true_label=test_label.copy()
from sklearn.metrics import roc_auc_score

reshaped_test_LSTM = test_LSTM.flatten()
reshaped_true_label = true_label.flatten()

# 计算 ROC AUC
roc_auc_LSTM = roc_auc_score(reshaped_true_label, reshaped_test_LSTM)
print(f"ROC AUC: {roc_auc_LSTM}")
from sklearn.metrics import precision_recall_curve, auc,f1_score
precision_LSTM, recall_LSTM, thresholds_LSTM = precision_recall_curve(reshaped_true_label, reshaped_test_LSTM)
# 计算 PR AUC
pr_auc_LSTM = auc(recall_LSTM, precision_LSTM)
print("PR AUC:", pr_auc_LSTM)
fpr_LSTM, tpr_LSTM, thresholds_LSTM = roc_curve(reshaped_true_label, reshaped_test_LSTM, drop_intermediate=False)
roc_auc_LSTM = auc(fpr_LSTM, tpr_LSTM)
print(roc_auc_LSTM)
true_test = test_LSTM.copy()
threshold = 0.5
reshaped_test_LSTM = (reshaped_test_LSTM > threshold).astype(int)
f1 = f1_score(reshaped_true_label, reshaped_test_LSTM)
print(f1)
true_label=test_label.copy()

# # RNN
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GRU, Bidirectional, Dense, Dropout, Attention,SimpleRNN
def get_model(max_len, feature_num):
    
    clear_session()
    inputs = Input(shape=(max_len, feature_num), dtype='float32')
    x = Dense(32, activation='relu')(inputs)
    x = SimpleRNN(64, return_sequences=True,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = SimpleRNN(64, return_sequences=True,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = SimpleRNN(64, return_sequences=True,activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.summary()
    return model

model_RNN = get_model(max_len=maxlen, feature_num=len(features))
adam = optimizers.Adam(lr=0.001)
loss = BinaryCrossentropy(from_logits=False)
k_auc = AUC()

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH, 
    monitor='val_auc', 
    mode='max', 
    save_best_only=True
)

model_RNN.compile(
    optimizer=adam, 
    loss=loss,
    metrics=[k_auc]
)

to_save['lr'] = 0.001
time_start = datetime.now()
history = model_RNN.fit(
    train_data, train_label,
    validation_data=(val_data, val_label), 
    batch_size=2048, 
    epochs=1, 
    callbacks=[checkpoint])
train_time = (datetime.now() - time_start).total_seconds()
to_save['train_time'] = train_time
print(train_time)
fpr_RNN, tpr_RNN, thresholds_our = roc_curve(reshaped_true_label, reshaped_test_RNN)
roc_auc_RNN = auc(fpr_RNN, tpr_RNN)
print(roc_auc_RNN)
from sklearn.metrics import precision_recall_curve, auc,f1_score
precision_RNN, recall_RNN, thresholds = precision_recall_curve(reshaped_true_label, reshaped_test_RNN)
# 计算 PR AUC,f1
pr_auc_RNN= auc(recall_RNN, precision_RNN)
print("PR AUC:", pr_auc_RNN)
threshold = 0.5
reshaped_test_RNN = (reshaped_test_RNN > threshold).astype(int)
f1 = f1_score(reshaped_true_label, reshaped_test_RNN)
print(f1)


# # GRU
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GRU, Bidirectional, Dense, Dropout, Attention
def get_model(max_len, feature_num):
    
    clear_session()
    inputs = Input(shape=(max_len, feature_num), dtype='float32')
    x = Dense(64, activation='relu')(inputs)
    x = GRU(64, return_sequences=True,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = GRU(64, return_sequences=True,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = GRU(64, return_sequences=True,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(2, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.summary()
    return model
model_GRU = get_model(max_len=maxlen, feature_num=len(features))
adam = optimizers.Adam(lr=0.001)
loss = BinaryCrossentropy(from_logits=False)
k_auc = AUC()

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH, 
    monitor='val_auc', 
    mode='max', 
    save_best_only=True
)

model_GRU.compile(
    optimizer=adam, 
    loss=loss,
    metrics=[k_auc]
)

to_save['lr'] = 0.001
time_start = datetime.now()
history = model_GRU.fit(
    train_data, train_label,
    validation_data=(val_data, val_label), 
    batch_size=2048, 
    epochs=1, 
    callbacks=[checkpoint])
train_time = (datetime.now() - time_start).total_seconds()
to_save['train_time'] = train_time
print(train_time)
test_GRU = model_GRU.predict(test_data)
from sklearn.metrics import roc_auc_score

# 将三维数组转换为一维数组
reshaped_test_GRU = test_GRU.flatten()
#reshaped_true_label = true_label.flatten()

# 计算 ROC AUC
roc_auc_GRU = roc_auc_score(reshaped_true_label, reshaped_test_GRU)
print(f"ROC AUC: {roc_auc_GRU}")
fpr_GRU, tpr_GRU, thresholds_GRU = roc_curve(reshaped_true_label, reshaped_test_GRU)
roc_auc_GRU = auc(fpr_GRU, tpr_GRU)
print(roc_auc_GRU)
from sklearn.metrics import precision_recall_curve, auc,f1_score
precision_GRU, recall_GRU, thresholds = precision_recall_curve(reshaped_true_label, reshaped_test_GRU)
# 计算 PR AUC
pr_auc_GRU= auc(recall_GRU, precision_GRU)
print("PR AUC:", pr_auc_GRU)
threshold = 0.5
reshaped_test_GRU = (reshaped_test_GRU > threshold).astype(int)
f1 = f1_score(reshaped_true_label, reshaped_test_GRU)
print(f1)

