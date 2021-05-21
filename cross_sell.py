import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import NMF
import sklearn
import tensorflow as tf
import os
import warnings
import pickle
warnings.filterwarnings("ignore")
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split
import time


def preprocess():
    # does preprocessing and create training and validation datasets
    start = time.time()
    df = pd.read_csv('Data_csv_file_utf.csv')
    df = df.drop(['Unnamed: 0'],axis=1)
    print(time.time()-start)    #0.36
    df['Doc. Date'] = pd.to_datetime(df['Doc. Date'])
    df['per_order'] = df['PCS delivered']/df['Order qty']
    df = df[df['Order qty'].apply(lambda x:False if int(x)!=x else True)]

    df_new = []
    grouped = df.groupby('Material')
    for a,b in grouped:
        b['per_order'] = b['per_order'].replace(0,np.nan)
        b = b.sort_values(by=['Order qty','per_order'])
        b['per_order'].fillna(method='ffill',inplace=True)
        b['per_order'].fillna(b['Order qty'],inplace=True)
        b['PCS delivered'] = b['Order qty']*b['per_order']
        b['hl_ratio'] = round(b['PCS delivered']/b['HL delivered'],5)
        val = b['hl_ratio'].unique()[0]
        if val==float('inf'):
            val = 16.66667
        b['hl_ratio'].replace(float('inf'),np.nan, inplace=True)
        b['hl_ratio'].fillna(val)
        b['HL delivered'] = b['PCS delivered']/val
        try:
            val2 = b['TTC'].dropna().unique()[0]
            b['TTC'] = val2
        except:
            b['TTC'] = -1
        try:
            val2 = b['MACO/HL '].dropna().unique()[0]
            b['MACO/HL '] = val2
        except:
            b['MACO/HL '] = -1
        df_new.append(b)
    df_new = pd.concat(df_new).sort_values(by=['Material','Order qty']).reset_index(drop=True)
    df_new = df_new[df_new['HL delivered']<200]
    df_new = df_new[df_new['delivery_flag']==1]


    
    os.makedirs('preprocessed_data/', exist_ok=True)
    np.save('preprocessed_data/users_unique.npy',df_new['Ship-to nu'].unique())
    print(time.time()-start)    #1.57

    df_fully_cleaned = df_new.dropna(subset=['Brand'])
    item_details = []
    for a, b in df_fully_cleaned.groupby('Material'):
        item = []
        item.append(a)
        item.append(b['Brand'].unique()[0])
        item.append(b['Subrand'].unique()[0])
        item.append(b['SEGMENTS : Pils / Spécialités / Superspécialités/Bouteille Young adult'].unique()[0])
        item.append(b['Container Type'].unique()[0])
        item.append(b['Container Size'].unique()[0])
        item.append(b['Variétés'].unique()[0])
        item.append(b['Segment LE'].unique()[0])
        item_details.append(item)
    item_details = pd.DataFrame(item_details, columns = ['item', 'brand', 'subrand', 'segments','cont_type','cont_size','varietes','segment_le'])
    item_details.to_csv('preprocessed_data/item_details.csv',index=False)

    popularity = df_new.groupby('Material')['Ship-to nu'].count().to_frame().reset_index(drop=False)
    popularity.columns = ['item','user']
    popularity.to_csv('preprocessed_data/popularity.csv',index=False)
    
    df_new = df_new.sort_values(by=['Ship-to nu','Material'])
    temp = df_new.copy()
    enc1 = LabelEncoder()
    x = temp.loc[:,'Ship-to nu']
    x_enc = enc1.fit_transform(x)
    temp.loc[:,'Ship-to nu'] = x_enc
    np.save('preprocessed_data/user_encoding.npy',enc1.classes_)
    enc2 = LabelEncoder()
    x = temp.loc[:,'Material']
    x_enc = enc2.fit_transform(x)
    temp.loc[:,'Material'] = x_enc
    np.save('preprocessed_data/item_encoding.npy',enc2.classes_)
    print(time.time()-start)    #1.69

    users = temp.groupby('Ship-to nu')['Material'].count().to_frame().reset_index()
    users['group'] = pd.qcut(users['Material'],18,labels=range(18))
    skf = StratifiedKFold(n_splits=5)
    X = users.loc[:,['Ship-to nu']]
    Y = users.loc[:,['group']]
    for i, (train_index, test_index) in enumerate(skf.split(X,Y)):
        users.loc[test_index,'fold'] = i
    user_items = pd.merge(temp[['Doc. Date','Material','Ship-to nu','HL delivered']], users[['Ship-to nu','fold']], how='left', on = 'Ship-to nu')
    user_items.columns = ['date','item','user','amount','fold']
    print(time.time()-start)    #1.81

    user_items = user_items.sort_values(by=['user','item','date'])
    user_items['date_shift'] = user_items['date'].shift(-1)
    grouped = user_items.groupby(['user','item'])
    stack = []
    max_date = user_items['date'].max()
    print(time.time()-start)    #1.85
    for a,b in grouped:
        b.loc[b.index[-1],'date_shift'] = max_date
        stack.append(b)
    print(time.time()-start)    #12.4
    user_items = pd.concat(stack,axis=0)
    user_items['diff'] =  (user_items['date_shift']-user_items['date']).dt.days
    user_items['diff'].replace(0,1,inplace=True)
    user_items.drop(labels=['date_shift'],axis=1, inplace=True)
    user_items['rating'] = user_items['amount']/user_items['diff']
    user_items = user_items.groupby(['user','item'])[['rating','fold']].mean().reset_index()
    print(time.time()-start)    #16.5

    unique_users = len(user_items['user'].unique())
    unique_items = len(user_items['item'].unique())
    train_ui = user_items[user_items['fold']!=4]
    val_full = user_items[user_items['fold']==4]
    val_tr, val_te = [],[]
    for a,b in val_full.groupby('user'):
        b = b.reset_index(drop=True)
        max_i = int(0.2*b.shape[0])
        if max_i!=0:
            val_te.append(b.loc[:max_i,:])
        val_tr.append(b.loc[max_i:,:])
    val_tr = pd.concat(val_tr, axis=0).reset_index(drop=True)
    val_te = pd.concat(val_te, axis=0).reset_index(drop=True)

    print(time.time()-start)    #16.6
    train_full = pd.concat([train_ui, val_tr], axis=0).reset_index(drop=True)
    temp = []
    train_min_max = []
    for a,b in train_full.groupby('user'):
        min_val, max_val = b['rating'].min(), b['rating'].max()
        if min_val==max_val:
            b['rating']=1.5
        else:
            b['rating'] = (b['rating']/max_val)
        temp.append(b)
        train_min_max.append((a,min_val,max_val))
    train_full = pd.concat(temp,axis=0).reset_index(drop=True)
    train_min_max = pd.DataFrame(train_min_max,columns=['user','min','max'])
    train_min_max.set_index('user',inplace=True)
    temp = []
    for a,b in val_te.groupby('user'):
        min_val, max_val = train_min_max.loc[a,'min'], train_min_max.loc[a,'max']
        if min_val==max_val:
            b['rating']=1.5
        else:
            b['rating'] = (b['rating']/max_val).clip(0,1)
        temp.append(b)
    val_te = pd.concat(temp,axis=0).reset_index(drop=True)

    assert(unique_users==178 and unique_items==176)
    ui_matrix = np.zeros(shape=(unique_users,unique_items))
    for i,row in train_full.iterrows():
        u = int(row['user'])
        i = int(row['item'])
        ui_matrix[u,i] = row['rating']
    check = np.where(ui_matrix!=0,1,0)
    ui = np.where(ui_matrix!=0,ui_matrix,-1)

    train_full.to_csv('preprocessed_data/train_full.csv',index=False)
    val_te.to_csv('preprocessed_data/val_te.csv',index=False)
    np.save('preprocessed_data/ui.npy',ui)
    train_min_max.to_csv('preprocessed_data/train_min_max.csv')
    print(time.time()-start)    #17.8
    return ui, train_min_max
