#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import psycopg2
import getpass
import gc
from tqdm import tqdm
from collections import Counter
user = 'postgres'
host = '10.10.116.166'
#127.0.0.1
#host = '127.0.0.1'
port = 8088
dbname = 'mimiciv'
schema = 'public, mimiciv_derived, mimiciv_hosp, mimiciv_icu'
con = psycopg2.connect(user=user, host=host, port=port,
                      dbname=dbname, password=getpass.getpass())
cur = con.cursor()
DATA_PATH = './data_rsmp6.tsv'
RESAMPLE_INTERVAL = '6H'
OUTPUT_FILENAME = './data.tsv'

# # add ventilation, sedatives and vasopressors
data = pd.read_csv(DATA_PATH, sep='\t')
data.head()
data.charttime = pd.to_datetime(data.charttime)
icustays = data.stay_id.unique().tolist()
len(icustays)

# 数据库查询语句
ventilation = 'select * from mimiciv_derived.ventilation'
sedative = 'select * from sedative'
vasopressor = 'select * from mimiciv_derived.vasopressin'
rrt = 'select * from rrt_all_admission_0721'
crrt = 'select * from mimiciv_derived.crrt'

query_dict = {
    'ventilation': ventilation,
    'sedative' : sedative
    'vasopressor': vasopressor,
    'rrt': rrt,
    'crrt': crrt,
}
for name, query in tqdm(query_dict.items(), ncols=100):
    table = pd.read_sql_query(query, con)
    table = table[table.stay_id.isin(icustays)]
    if pd.isna(table).any().any():
        print('{0} has None!'.format(name))
        continue
    table = table[['stay_id', 'starttime', 'endtime']]
    table['starttime'] = pd.to_datetime(table['starttime'])
    table['endtime'] = pd.to_datetime(table['endtime'])
    table.drop_duplicates(inplace=True)
    table.sort_values(by=['stay_id', 'starttime'], inplace=True, ascending=True, ignore_index=True)
    
    table_group = table.groupby('stay_id')
    data_group = data.copy()
    data_group[name] = 0
    data_group = data_group.set_index('charttime').groupby('stay_id')
    
    result_list = []
    for i in icustays:
        tmp_data = data_group.get_group(i).copy()
        try:
            tmp_table = table_group.get_group(i).copy()
            for idx in tmp_table.index:
                starttime, endtime = tmp_table.loc[idx, 'starttime'], tmp_table.loc[idx, 'endtime']
                tmp_data.loc[starttime:endtime, name] = 1
        except:
            pass
        result_list.append(tmp_data.reset_index(drop=False))
    data = pd.concat(result_list, axis=0, ignore_index=True)
#     print('{0} first: {1}'.format(name, Counter(data[name])))
    
    data_group = data.groupby('stay_id')
    table_start = table[['stay_id', 'starttime']]
    table_end = table[['stay_id', 'endtime']]
    table_end.columns = ['stay_id', 'starttime']
    table = pd.concat([table_start, table_end], axis=0)
    table.sort_values(by=['stay_id', 'starttime'], axis=0, ascending=True, inplace=True)
    table.drop_duplicates(inplace=True)
    table_group = table.set_index('starttime').groupby('stay_id')
    
    result_list = []
    for i in icustays:
        tmp_data = data_group.get_group(i).copy()
        try:
            tmp_table = table_group.get_group(i).copy()
            for idx in tmp_data.index:
                starttime = tmp_data.loc[idx, 'charttime']
                endtime = starttime + pd.Timedelta(RESAMPLE_INTERVAL) - pd.Timedelta('1s')
                if len(tmp_table.loc[starttime:endtime, ]) == 0:
                    pass
                else:
                    tmp_data.loc[idx, name] = 1
        except:
            pass
        result_list.append(tmp_data)
    data = pd.concat(result_list, axis=0, ignore_index=True)
#     print('{0} second: {1}'.format(name, Counter(data[name])))
    
    del table, table_start, table_end, table_group, tmp_data, tmp_table, data_group, result_list
    gc.collect()

data.columns

# 数据库查询语句
#ventilation = 'select * from ventilation'
ventilation = 'select * from mimiciv_derived.ventilation'
sedative = 'select * from sedative'
vasopressor = 'select * from mimiciv_derived.vasopressin'
rrt = 'select * from rrt_all_admission_0721'
crrt = 'select * from mimiciv_derived.crrt'
#adenosine = 'select * from adenosine'
#isuprel = 'select * from isuprel'
query_dict = {
    #'ventilation': ventilation,
    'sedative': sedative,
    #'vasopressor': vasopressor,
    #'rrt': rrt,
    #'crrt': crrt,
    #'adenosine': adenosine,
    #'isuprel': isuprel
}
for name, query in tqdm(query_dict.items(), ncols=100):
    table = pd.read_sql_query(query, con)
    table = table[table.stay_id.isin(icustays)]
    if pd.isna(table).any().any():
        print('{0} has None!'.format(name))
        continue
    table = table[['stay_id', 'starttime', 'endtime']]
    table['starttime'] = pd.to_datetime(table['starttime'])
    table['endtime'] = pd.to_datetime(table['endtime'])
    table.drop_duplicates(inplace=True)
    table.sort_values(by=['stay_id', 'starttime'], inplace=True, ascending=True, ignore_index=True)
    
    table_group = table.groupby('stay_id')
    data_group = data.copy()
    data_group[name] = 0
    data_group = data_group.set_index('charttime').groupby('stay_id')
    
    result_list = []
    for i in icustays:
        tmp_data = data_group.get_group(i).copy()
        try:
            tmp_table = table_group.get_group(i).copy()
            for idx in tmp_table.index:
                starttime, endtime = tmp_table.loc[idx, 'starttime'], tmp_table.loc[idx, 'endtime']
                tmp_data.loc[starttime:endtime, name] = 1
        except:
            pass
        result_list.append(tmp_data.reset_index(drop=False))
    data = pd.concat(result_list, axis=0, ignore_index=True)
#     print('{0} first: {1}'.format(name, Counter(data[name])))
    
    data_group = data.groupby('stay_id')
    table_start = table[['stay_id', 'starttime']]
    table_end = table[['stay_id', 'endtime']]
    table_end.columns = ['stay_id', 'starttime']
    table = pd.concat([table_start, table_end], axis=0)
    table.sort_values(by=['stay_id', 'starttime'], axis=0, ascending=True, inplace=True)
    table.drop_duplicates(inplace=True)
    table_group = table.set_index('starttime').groupby('stay_id')
    
    result_list = []
    for i in icustays:
        tmp_data = data_group.get_group(i).copy()
        try:
            tmp_table = table_group.get_group(i).copy()
            for idx in tmp_data.index:
                starttime = tmp_data.loc[idx, 'charttime']
                endtime = starttime + pd.Timedelta(RESAMPLE_INTERVAL) - pd.Timedelta('1s')
                if len(tmp_table.loc[starttime:endtime, ]) == 0:
                    pass
                else:
                    tmp_data.loc[idx, name] = 1
        except:
            pass
        result_list.append(tmp_data)
    data = pd.concat(result_list, axis=0, ignore_index=True)
#     print('{0} second: {1}'.format(name, Counter(data[name])))
    
    del table, table_start, table_end, table_group, tmp_data, tmp_table, data_group, result_list
    gc.collect()

data.columns

# add time
details = 'select * from icustay_detail_0721'
details = pd.read_sql_query(details, con)
details.head()
details.drop(['hadm_id', 'subject_id', 'admittime', 'dischtime'], axis=1, inplace=True)
icustays = data.stay_id.unique().tolist()

details = details[details.stay_id.isin(icustays)]
details.head()
details = details[details.stay_id.isin(icustays)]
details.head()
details.dtypes
details.stay_id.nunique()
details.gender.nunique(), details.race.nunique(), details.admission_type.nunique()

details.race.replace({'ASIAN - ASIAN INDIAN': 'ASIAN', 'ASIAN - CHINESE': 'ASIAN',
                      'ASIAN - KOREAN': 'ASIAN', 'ASIAN - SOUTH EAST ASIAN': 'ASIAN',
                      'BLACK/AFRICAN AMERICAN': 'BLACK', 'BLACK/CARIBBEAN ISLAND': 'BLACK',
                      'BLACK/AFRICAN': 'BLACK', 'BLACK/AFRICAN AMERICAN': 'BLACK',
                      'HISPANIC OR LATINO': 'HISPANIC/LATINO',
                      'HISPANIC/LATINO - CENTRAL AMERICAN': 'HISPANIC/LATINO',
                      'HISPANIC/LATINO - COLUMBIAN': 'HISPANIC/LATINO', 'HISPANIC/LATINO - CUBAN': 'HISPANIC/LATINO',
                      'HISPANIC/LATINO - DOMINICAN': 'HISPANIC/LATINO', 'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC/LATINO',
                      'HISPANIC/LATINO - HONDURAN': 'HISPANIC/LATINO', 'HISPANIC/LATINO - MEXICAN': 'HISPANIC/LATINO',
                      'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC/LATINO',
                      'HISPANIC/LATINO - SALVADORAN': 'HISPANIC/LATINO',
                      'WHITE - BRAZILIAN': 'WHITE', 'WHITE - EASTERN EUROPEAN': 'WHITE',
                      'WHITE - OTHER EUROPEAN': 'WHITE', 'WHITE - RUSSIAN': 'WHITE',
                      'UNABLE TO OBTAIN': 'UNKNOWN', 'PATIENT DECLINED TO ANSWER': 'UNKNOWN',
                      'OTHER': 'UNKNOWN','BLACK/CAPE VERDEAN': 'BLACK'
                     }, inplace=True)

details.gender.nunique(), details.race.nunique(), details.admission_type.nunique()
Counter(details.gender.values)

details.admission_age.describe()

details = pd.get_dummies(details).astype('float32')

details.shape

details.head()

data = pd.merge(data, details, how='left', on='stay_id')

data.columns

data.columns

aki = data.aki_stage.unique().tolist()
len(aki)


data.to_csv(OUTPUT_FILENAME, sep='\t', index=False)


cur.close()
con.close()



