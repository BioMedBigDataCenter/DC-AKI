#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import psycopg2
import getpass
import gc
from tqdm import tqdm

user = 'postgres'
host = '10.10.116.166'
port = 8088
dbname = 'mimiciv'
schema = 'public, mimiciv_derived, mimiciv_hosp, mimiciv_icu'

# 连接本地数据库
con = psycopg2.connect(user=user, host=host, port=port,
                      dbname=dbname, password=getpass.getpass())
cur = con.cursor()

OUTPUT_FILENAME = './data_seq_ori.tsv'
VITALS = 'select * from vital_all_icu_0721'
LABS = 'select * from labs_all_icu_mean_0721'
KDIGO_STAGES = 'select * from kdigo_stages_0721'


# # vitals
vitals = pd.read_sql_query(VITALS, con)
vitals.drop(['subject_id'], axis=1, inplace=True)
vitals.head()
vitals.shape, vitals.dtypes
vitals.hadm_id.nunique(), vitals.stay_id.nunique()
vitals.columns = ['hadm_id', 'stay_id', 'charttime', 'heartrate', 'sysbp', 'diasbp', 'meanbp', 'resprate', 'tempc', 'spo2', 'glucose']
vitals.dropna(subset=vitals.columns[3:], how='all', inplace=True)
vitals.sort_values(['stay_id', 'charttime'], inplace=True, ascending=True)
vitals.shape, pd.isna(vitals).any()
vitals.hadm_id.nunique(), vitals.stay_id.nunique()


# # labs
labs = pd.read_sql_query(LABS, con)
labs.drop(['subject_id'], axis=1, inplace=True)
labs.head()
labs.shape, labs.dtypes

labs.hadm_id.nunique(), labs.stay_id.nunique()
labs.columns = ['hadm_id', 'stay_id', 'charttime', 'aniongap', 'albumin', 'bands', 'bicarbonate', 
                'bilirubin', 'creatinine', 'chloride', 'glucose', 'hematocrit', 'hemoglobin', 'lactate', 
                'platelet', 'potassium', 'ptt', 'inr', 'pt', 'sodium', 'bun', 'wbc']
labs.dropna(subset=labs.columns[3:], how='all', inplace=True)
labs.sort_values(['stay_id', 'charttime'], inplace=True, ascending=True)
labs.shape, pd.isna(labs).any()
labs.hadm_id.nunique(), labs.stay_id.nunique()


# # kdigo stages
kdigo_stages = pd.read_sql_query(KDIGO_STAGES, con)
kdigo_stages.drop(['subject_id', 'icu_intime', 'icu_outtime'], axis=1, inplace=True)
kdigo_stages.head()
kdigo_stages.shape, kdigo_stages.dtypes
kdigo_stages.dropna(subset=kdigo_stages.columns[3:], how='all', inplace=True)
kdigo_stages.shape, pd.isna(kdigo_stages).any()
kdigo_stages.hadm_id.nunique(), kdigo_stages.stay_id.nunique()
glucose = vitals[['hadm_id', 'stay_id', 'charttime', 'glucose']].copy()
glucose.dropna(subset=['glucose'], inplace=True)
glucose_lab = labs[['hadm_id', 'stay_id', 'charttime', 'glucose']].copy()
glucose_lab.dropna(subset=['glucose'], inplace=True)
glucose.shape, glucose_lab.shape
glucose = glucose.append(glucose_lab, ignore_index=True)
glucose.drop_duplicates(keep='first', inplace=True)
glucose.shape
vitals.drop(['glucose'], axis=1, inplace=True)
vitals.dropna(subset=vitals.columns[3:], how='all', inplace=True)
labs.drop(['glucose'], axis=1, inplace=True)
labs.dropna(subset=labs.columns[3:], how='all', inplace=True)

scr = kdigo_stages[['hadm_id', 'stay_id', 'charttime', 'creat']].copy()
scr.dropna(subset=['creat'], inplace=True)
scr_lab = labs[['hadm_id', 'stay_id', 'charttime', 'creatinine']].copy()
scr_lab.dropna(subset=['creatinine'], inplace=True)
scr.shape, scr_lab.shape
scr.rename(columns={'creat': 'creatinine'}, inplace=True)
scr = scr.append(scr_lab, ignore_index=True)
scr.drop_duplicates(keep='first', inplace=True)
scr.shape
kdigo_stages.drop(['creat'], axis=1, inplace=True)
kdigo_stages.dropna(subset=kdigo_stages.columns[3:], how='all', inplace=True)
labs.drop(['creatinine'], axis=1, inplace=True)
labs.dropna(subset=labs.columns[3:], how='all', inplace=True)
merge_axis = ['hadm_id', 'stay_id', 'charttime']
data = pd.merge(vitals, labs, on=merge_axis, how='outer')
data = pd.merge(data, glucose, on=merge_axis, how='outer')
data = pd.merge(data, scr, on=merge_axis, how='outer')
data = pd.merge(data, kdigo_stages, on=merge_axis, how='outer')
del vitals, labs, glucose, glucose_lab, scr, scr_lab, kdigo_stages
gc.collect()
data.columns
data.shape
data.hadm_id.nunique(), data.stay_id.nunique()
pd.isna(data[data.columns[:3]]).any()
data.sort_values(['hadm_id', 'stay_id', 'charttime'], inplace=True, ascending=True, ignore_index=True)
data.to_csv(OUTPUT_FILENAME, sep='\t', index=False)
cur.close()
con.close()

