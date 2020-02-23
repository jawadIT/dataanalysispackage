# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:52:32 2020

@author: jawad_zf1uaw5
"""

import pandas as pd
import numpy as np
import json as js
from pyxlsb import open_workbook as open_xlsb

import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Input
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

RANDOM_SEED = 314
TEST_PCT = 0.2


def get_file(path,filename,filetype):
    
    if filetype == 'csv':
        try:
            return(readcsv(path,filename))
        except:
            print('error in reading csv')
    else:
        if filetype == 'json':
            try:
                with open(f'{path}\\{filename}', 'r') as f:
                    return(pd.DataFrame(js.load(f)))
            except:
                print('error in reading json')
        else:
            if filetype == 'xlsb':
                try:
                    print('reading file...')
                    df = []
                    with open_xlsb(f'{path}\\{filename}') as wb:
                        with wb.get_sheet('Data') as sheet:
                            for row in sheet.rows():
                                df.append([item.v for item in row])
                    print('reading completed ...')
                    return(pd.DataFrame(df[1:], columns=df[0]))
                except:
                    print('error in reading xlsb')
            else:
                print('file format not supported yet in packages')
        

def readcsv(path,filename):
    
    data = []
    for chunk in pd.read_csv(f'{path}\\{filename}',low_memory=False,chunksize=20000):
        data.append(chunk)
    
    return pd.concat(data,axis=0)

def check_df_missing(df,keys):
    
    df.columns = [c.lower() for c in df.columns]   
    
    colms = list(df.columns)
    for key in keys:
        colms.remove(key)
    
    cols = []
    nullcount = []
    nullpercentage = []
    for c in df.columns:
        cols.append(c)
        nullcount.append(df[c].isnull().sum())
        nullpercentage.append((df[c].isnull().sum()*100)/df.shape[0])
    df_col_info = pd.DataFrame()
    df_col_info['col'] = cols
    df_col_info['null_count'] = nullcount
    df_col_info['null_per'] = nullpercentage
    
    df['non_null_count'] = df[colms].apply(lambda x: x.count() , axis = 1)
    df_row_info = df[df['non_null_count'] == 0].copy()
    
    #all_null_cols = df.columns[df.isnull().all()]
    #null_cols = df.columns[df.isnull().any()]
    
    return df_col_info,pd.DataFrame(df_row_info)

def clean_df_missing(df,df_row_info,cols):
        
    df = df.drop(cols,axis=1)
    
    if df_row_info.shape[0] > 0:
        df = df[~df.index.isin(df_row_info.index)]
    else:
        df.fillna(method='ffill',inplace=True)
    
    #df_all = df.merge(df_row_info.drop_duplicates(), on=[ind],how='left', indicator=True)
    #df = df_all[df_all['_merge']== 'left_only']   
    
    
    return df

def plot_freq(df,keys):
    
    col_names = list(df.columns)
    for key in keys:
        col_names.remove(key)
    
    fig, ax = plt.subplots(len(col_names), figsize=(16,12))
    
    for i, col_val in enumerate(col_names):
    
        sns.distplot(df[col_val], hist=True, ax=ax[i])
        ax[i].set_title('Freq dist: '+col_val, fontsize=10)
        ax[i].set_xlabel(col_val, fontsize=8)
        ax[i].set_ylabel('Count', fontsize=8)
    
    return plt


def shift_observations(labelcol,shiftby,df):
    
    sign = lambda x: (1, -1)[x < 0]

    vector = df[labelcol].copy()
    
    for s in range(abs(shiftby)):
        tmp = vector.shift(sign(shiftby))
        tmp = tmp.fillna(0)
        vector += tmp

    df.insert(loc=0, column=labelcol+'tmp', value=vector)
    df = df.drop(df[df[labelcol] == 1].index)
    df = df.drop(labelcol, axis=1)
    df = df.rename(columns={labelcol+'tmp': labelcol})
    df.loc[df[labelcol] > 0, labelcol] = 1
    
    return df

def lookbackintime(input_X,input_y,lookback):
    
    output_X = []
    output_y = []
    for i in range(len(input_X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            t.append(input_X[[(i + j + 1)], :])
        output_X.append(t)
        output_y.append(input_y[i + lookback + 1])
    
    return np.squeeze(np.array(output_X)), np.array(output_y)
    
def to_2D(df):
    
    df2d = np.empty((df.shape[0], df.shape[2]))

    for i in range(df.shape[0]):
            df2d[i] = df[i, (df.shape[1]-1), :]
    
    return df2d
    
def to_3D(df,lookback,n_features):
    
    df3d = df.reshape(df.shape[0], lookback, n_features)
    
    return df3d
    
def scale(df,Scaler,features):
    
    scaler = Scaler.fit(df[features])
    dfscaled = pd.concat([df,pd.DataFrame(scaler.transform(df[features]),columns=features)],axis=1)
    
    return dfscaled,scaler

def autoencoder(x,btsize,ep,activfunc):
    
    indim = x.shape[1]    
    input_layer = Input(shape=(indim,))
    encoder = Dense(12,kernel_initializer='normal',activation=activfunc)(input_layer)
    
    nodes = []
    nodecount = indim * 4      
    while (nodecount)/2 >=2:        
        nodes.append(nodecount)
        nodecount = round(nodecount/2)
    nodes.append(nodecount)
    
    for node in nodes:
        encoder = Dense(node,activation=activfunc)(encoder)
    
    nodes.sort(reverse=False)
    nodes = nodes[1:]
    decoder = Dense(nodes[6],activation=activfunc)(encoder)
    
    nodes = nodes[1:]
    for node in nodes:
        decoder = Dense(node,activation=activfunc)(decoder)
        
    decoder = Dense(indim,activation=activfunc)(decoder)
        
    model = Model(inputs=input_layer, outputs = decoder)
    
    return model



