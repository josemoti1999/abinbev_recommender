import flask
from flask import render_template, url_for, redirect, session
import os
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from cs_train import train, predict
from other_details import get_item_user_details, get_most_popular, get_similar_items
from upsell import upsell_recommendation
from csv import writer
from datetime import date
import time

app = flask.Flask(__name__, template_folder='templates')
app.secret_key = "super secret key"
session = {}
new_sale_count = 0

@app.route('/added_display', methods=['GET','POST'])
def added_display():
    global new_sale_count
    user_code_list = list(np.sort(np.load('preprocessed_data/users_unique.npy').astype(int)))
    item_code_list = list(np.sort(np.load('preprocessed_data/items_unique.npy').astype(int)))
    date_today = date.today()
    date_today = date_today.strftime("%Y-%m-%d")
    if flask.request.method == 'GET':
        return flask.render_template('added_display.html',user_code_list=user_code_list,
         item_code_list=item_code_list, date_today=date_today)
    if flask.request.method == 'POST':
        new_sale_count+=1
        hl = flask.request.form['hl']
        user_code = flask.request.form['user_code']
        item_code = flask.request.form['item_code']
        new_row = [str(date_today)+' 00:00:00',int(item_code), 1,int(user_code),20,float(hl),'',1]
        with open('Data_additional.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(new_row)
            f_object.close()
        return flask.render_template('added_display.html',user_code_list=user_code_list,
         item_code_list=item_code_list, date_today=date_today,post_bool=True)


@app.route('/item_desc/<user_code>/<item_code>', methods=['GET','POST'])
def item_desc(user_code,item_code):
    user_code = int(user_code)
    item_code = int(item_code)
    date_today = date.today()
    date_today = date_today.strftime("%Y-%m-%d")
    details = session['details'][item_code]
    item_details = []
    user_details = []
    dict_1, dict_2 = get_item_user_details(item_code, user_code)
    for key, val in dict_1: item_details.append(f'{key} --> {val}')
    for key, val in dict_2: user_details.append(f'{key} --> {val}')
    if len(item_details)==0:    item_details.append('Item details not given')
    if len(user_details)==0:    user_details.append('User details not given')
    global new_sale_count

    if flask.request.method == 'GET':
        return flask.render_template('item_desc.html',user_code=user_code,date_today=date_today,item_code=item_code,details=details, item_details=item_details, user_details=user_details)
    if flask.request.method=='POST':
        new_sale_count+=1
        hl = flask.request.form['hl']
        new_row = [str(date_today)+' 00:00:00',int(item_code), 1,int(user_code),20,float(hl),'',1]
        with open('Data_additional.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(new_row)
            f_object.close()
        return flask.render_template('item_desc.html',user_code=user_code,date_today=date_today,item_code=item_code,details=details, item_details=item_details, user_details=user_details, post_bool=True)



@app.route('/', methods=['GET', 'POST'])
def main():
    unique_users = list(np.sort(np.load('preprocessed_data/users_unique.npy').astype(int)))
    print(len(unique_users))
    global new_sale_count
    if flask.request.method == 'GET':
        print('Getting main')
        session.clear()
        return (flask.render_template('home2.html',unique_users=unique_users,new_sale_count=new_sale_count))
            
    if flask.request.method == 'POST':
        if flask.request.form['btn_identifier']=='retrainer':
            print('Retraining')
            train(retrain=True)
            new_sale_count=0
            return (flask.render_template('home2.html',unique_users=unique_users,new_sale_count=new_sale_count,flash_retrain=True))
        elif flask.request.form['btn_identifier']=='adder':
            print('Adding new data')
            return flask.redirect('added_display')
        else:
            user = flask.request.form['user_code']
            
            enc1 = LabelEncoder()
            b = np.load('preprocessed_data/user_encoding.npy')
            enc1.classes_ = b
            user_encoded = enc1.transform([int(user)])
            enc2 = LabelEncoder()
            b = np.load('preprocessed_data/item_encoding.npy')
            enc2.classes_ = b

            cs_rec, cs_rec_scores, ph_rec, ph_rec_scores = predict(user_encoded)
            cs_rec, ph_rec = enc2.inverse_transform(cs_rec), enc2.inverse_transform(ph_rec)

            pop_rec, pop_rec_scores = get_most_popular()
            sim_rec = get_similar_items(ph_rec[:10])
            try:
                sim_rec, sim_rec_scores = zip(*(sim_rec)) 
            except:
                sim_rec, sim_rec_scores = [],[]
            print(ph_rec)
            up_rec, up1, up_rec_scores, up2, up3 = upsell_recommendation(ph_rec)
            session['details'] = defaultdict(list)

            # cross selling recommendations
            for i,ele in enumerate(cs_rec):
                session['details'][int(ele)].append(f'Cross sell opportunity')
                session['details'][int(ele)].append(f'This product has never been bought before by the given user {user}.')
                session['details'][int(ele)].append(f'Cross-sell rating --> {round(cs_rec_scores[i],4)}, Cross Sell Ranking --> {i+1} out of {len(cs_rec)}')
            
            # up selling recommendations
            for i,ele in enumerate(up_rec):
                session['details'][int(ele)].append(f'')
                session['details'][int(ele)].append(f'Upsell opportunity')
                session['details'][int(ele)].append(f'Good upselling opportunity for the given user {user}.')
                session['details'][int(ele)].append(f'Similarity score of {round(up_rec_scores[i],4)} to item {up1[i]} which is one of top 10 favourite product of the user.')
                session['details'][int(ele)].append(f'Cost of this product is {round(up2[i],4)} and cost of item {up1[i]} is {round(up3[i])}. Hence upselling chance.')

            # product history based recommenddations
            for i,ele in enumerate(ph_rec):
                session['details'][int(ele)].append(f'')
                session['details'][int(ele)].append(f'User History')
                session['details'][int(ele)].append(f'This product has been bought multiple times before by the given user {user}.')
                session['details'][int(ele)].append(f'User history rating score --> {round(ph_rec_scores[i],4)}, Ranking --> {i+1} out of {len(ph_rec)}')

            # overall popularity recommenddations
            for i,ele in enumerate(pop_rec):
                session['details'][int(ele)].append(f'')
                session['details'][int(ele)].append(f'Popularity')
                if i<50:
                    session['details'][int(ele)].append(f'Largely popular product among AB-Inbev users.')
                else:
                    session['details'][int(ele)].append(f'Not so popular among AB-Inbev users.')
                session['details'][int(ele)].append(f'Popularity score --> {round(pop_rec_scores[i],4)}, Popularity ranking --> {i+1} out of {len(pop_rec)}')

            # content based similar products
            for i,ele in enumerate(sim_rec):
                session['details'][int(ele)].append(f'')
                session['details'][int(ele)].append(f'Content based..')
                if i<len(sim_rec)//2:
                    session['details'][int(ele)].append(f'Content matching with several frequently bought products')
                else:
                    session['details'][int(ele)].append(f'Not similar to frequently bought products')
                session['details'][int(ele)].append(f'Similarity score --> {round(sim_rec_scores[i],4)},Similarity Ranking --> {i+1} out of {len(sim_rec)}')



            return flask.render_template('visualize.html',cs_rec=cs_rec[:6],cs_rec_scores=cs_rec_scores[:6],
                                                        up_rec=up_rec, up_rec_scores=up_rec_scores,
                                                        ph_rec=ph_rec, ph_rec_scores=ph_rec_scores, 
                                                        pop_rec=pop_rec, pop_rec_scores=pop_rec_scores,
                                                        sim_rec=sim_rec, sim_rec_scores=sim_rec_scores,
                                                        user_code=user)

if __name__ == '__main__':
    if not os.path.exists('preprocessed_data/ui_nmf_pred.npy'):
        train()
    app.run(debug=True)