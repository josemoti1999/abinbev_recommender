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
from csv import writer
import random
random.seed(0)


def preprocess(retrain):

    # does preprocessing and create training and validation datasets
    start = time.time()
    df = pd.read_csv('Data_orig.csv')
    df = df.drop(['Unnamed: 0'],axis=1)
    if retrain==True:
        df_additional = pd.read_csv('Data_additional.csv')
        print('Adding new sales')
        df = pd.concat([df,df_additional], axis=0, ignore_index=True)
    df['Doc. Date'] = pd.to_datetime(df['Doc. Date'])
    df['Order qty'].fillna(1, inplace=True)
    df['delivery_flag'].fillna(1, inplace=True)
    df['PCS delivered'].fillna(20, inplace=True)
    df['per_order'] = df['PCS delivered']/df['Order qty']
    df = df[df['Order qty'].apply(lambda x:False if int(x)!=x else True)]
    print(time.time()-start)    #0.44

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
        b.loc[b['HL delivered']==0,'HL delivered'] = b['PCS delivered']/val
        item_feats = ['MACO/HL ','Brand','Subrand','SEGMENTS : Pils / Spécialités / Superspécialités/Bouteille Young adult','Container Type','Container Size','Variétés','Segment LE','Degre Alc']
        for col_val in item_feats:
            try:
                val2 = b[col_val].dropna().unique()[0]
                b[col_val] = val2
            except:
                pass
        df_new.append(b)
    df_new = pd.concat(df_new).sort_values(by=['Material','Order qty']).reset_index(drop=True)
    temp = []
    grouped = df_new.groupby('Ship-to nu')
    for a,b in grouped:
        user_feats = ['Groupement','Postal Code','Street','Sous groupement','M2_Territory_ID','M1_Territory_ID','Dépt']
        for col_val in user_feats:
            try:
                val2 = b[col_val].dropna().unique()[0]
                b[col_val] = val2
            except:
                pass
        temp.append(b)
    df_new = pd.concat(temp).sort_values(by=['Material','Order qty']).reset_index(drop=True)
    df_new = df_new[df_new['HL delivered']<200]
    df_new = df_new[df_new['delivery_flag']==1]
    os.makedirs('preprocessed_data/', exist_ok=True)
    np.save('preprocessed_data/users_unique.npy',df_new['Ship-to nu'].unique())
    np.save('preprocessed_data/items_unique.npy',df_new['Material'].unique())
    print(time.time()-start)    #2.56

    cost_df = df_new.dropna(subset=item_feats[1:],how='any')
    gd = cost_df.groupby('Material')
    cost_sr = gd['MACO/HL '].mean()
    cost_df = pd.DataFrame(cost_sr)
    num_min = cost_df['MACO/HL '].min()
    num_max = cost_df['MACO/HL '].max()
    def cost(x):
        if not type(x) == 'int':
            return random.uniform(num_min, num_max)
    cost_df['MACO/HL '] = cost_df['MACO/HL '].apply(lambda x: cost(x))
    cost_df.to_csv('preprocessed_data/cost.csv')



    # checkpoints for training and predicting process
    df_fully_cleaned = df_new.dropna(subset=item_feats[1:],how='any')
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
        item.append(b['Degre Alc'].unique()[0])
        item_details.append(item)
    item_details = pd.DataFrame(item_details, columns = ['item', 'brand', 'subrand', 'segments','cont_type','cont_size','varietes','segment_le','degree_alc'])
    item_details.to_csv('preprocessed_data/item_details.csv',index=False)

    df_fully_cleaned = df_new.dropna(subset=user_feats,how='any')
    user_details = []
    for a, b in df_fully_cleaned.groupby('Ship-to nu'):
        user = []
        user.append(a)
        for f in user_feats:
            user.append(b[f].unique()[0])
        user_details.append(user)
    user_details = pd.DataFrame(user_details, columns = ['user','groupement','postal_code','street','Sous_groupement','m2_id','m1_id','dept'])
    user_details.to_csv('preprocessed_data/user_details.csv',index=False)

    popularity = df_new.groupby('Material')['Ship-to nu'].count().to_frame().reset_index(drop=False)
    popularity.columns = ['item','user']
    popularity.to_csv('preprocessed_data/popularity.csv',index=False)
    print(time.time()-start)    #2.65



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

    users = temp.groupby('Ship-to nu')['Material'].count().to_frame().reset_index()
    users['group'] = pd.qcut(users['Material'],18,labels=range(18))
    skf = StratifiedKFold(n_splits=5)
    X = users.loc[:,['Ship-to nu']]
    Y = users.loc[:,['group']]
    for i, (train_index, test_index) in enumerate(skf.split(X,Y)):
        users.loc[test_index,'fold'] = i
    user_items = pd.merge(temp[['Doc. Date','Material','Ship-to nu','HL delivered']], users[['Ship-to nu','fold']], how='left', on = 'Ship-to nu')
    user_items.columns = ['date','item','user','amount','fold']
    print(time.time()-start)    #2.707


    user_items = user_items.sort_values(by=['user','item','date'])
    user_items['date_shift'] = user_items['date'].shift(-1)
    grouped = user_items.groupby(['user','item'])
    stack = []
    max_date = user_items['date'].max()
    for a,b in grouped:
        b.loc[b.index[-1],'date_shift'] = max_date
        stack.append(b)
    user_items = pd.concat(stack,axis=0)
    user_items['diff'] =  (user_items['date_shift']-user_items['date']).dt.days
    user_items['diff'].replace(0,1,inplace=True)
    user_items.drop(labels=['date_shift'],axis=1, inplace=True)
    user_items['rating'] = user_items['amount']/user_items['diff']
    user_items = user_items.groupby(['user','item'])[['rating','fold']].mean().reset_index()
    print(time.time()-start)    #7.809

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
    print(time.time()-start)    #7.827

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
    print(time.time()-start)    #8.444
    print('Preprocessing done')
    return ui, train_min_max
