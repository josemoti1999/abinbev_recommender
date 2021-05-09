import flask
from flask import render_template, url_for, redirect, session
import os
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from cs_train import train, predict
from other_details import get_item_details, get_most_popular, get_similar_items

print('Preprocessing done')

app = flask.Flask(__name__, template_folder='templates')
app.secret_key = "super secret key"

@app.route('/item_desc/<user_code>/<item_code>', methods=['GET','POST'])
def item_desc(user_code,item_code):
    user_code = int(user_code)
    item_code = int(item_code)
    if flask.request.method == 'GET':
        details = session['details'][str(item_code)]
        
        return flask.render_template('item_desc.html',user_code=user_code,item_code=item_code,details=details)

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        session.clear()
        unique_users = list(np.sort(np.load('preprocessed_data/users_unique.npy').astype(int)))
        print(len(unique_users))
        return (flask.render_template('home2.html',unique_users=unique_users))
            
    if flask.request.method == 'POST':
        user = flask.request.form['user_code']
        if not os.path.exists('preprocessed_data/ui_nmf_pred.npy'):
            train()
        
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
        sim_rec = get_similar_items(ph_rec)
        sim_rec, sim_rec_scores = zip(*(sim_rec)) 

        session['details'] = defaultdict(list)
        seen = set()

        # cross selling recommendations
        for i,ele in enumerate(cs_rec):
            if ele not in seen:
                dict_ = get_item_details(ele)
                for key, val in dict_:
                    session['details'][int(ele)].append(f'1. {key} of the item is/are {val}')
                seen.add(int(ele))
            session['details'][int(ele)].append(f'2. This product has never been bought before.')
            session['details'][int(ele)].append(f'3. Predicted rating score of {cs_rec_scores[i]} and is {i+1}th ranked among cross-selling opportunities product recommended')


        # product history based recommenddations
        for i,ele in enumerate(ph_rec):
            if ele not in seen:
                dict_ = get_item_details(ele)
                for key, val in dict_:
                    session['details'][int(ele)].append(f'1. {key} of the item is/are {val}')
                seen.add(int(ele))
            session['details'][int(ele)].append(f'4. This product has been bought multiple times before')
            session['details'][int(ele)].append(f'5. Has a rating score of {ph_rec_scores[i]} according to our rating algorithm and is {i+1}th ranked among the items bought till now')

        # overall popularity recommenddations
        for i,ele in enumerate(pop_rec):
            if ele not in seen:
                dict_ = get_item_details(ele)
                for key, val in dict_:
                    session['details'][int(ele)].append(f'1. {key} of the item is/are {val}')
                seen.add(int(ele))
            session['details'][int(ele)].append(f'4. This product has been bought by a lot of different users before')
            session['details'][int(ele)].append(f'5. Has a rating score of {pop_rec_scores[i]} according to our rating algorithm and is {i+1}th ranked on the overall popularity')


        # similar products based on previous purchased
        for i,ele in enumerate(sim_rec):
            if ele not in seen:
                dict_ = get_item_details(ele)
                for key, val in dict_:
                    session['details'][int(ele)].append(f'1. {key} of the item is/are {val}')
                seen.add(int(ele))
            session['details'][int(ele)].append(f'4. This product is a lot similar to some of the users favourite product')
            session['details'][int(ele)].append(f'5. Has a rating score of {sim_rec_scores[i]} according to our rating algorithm and is {i+1}th ranked on the overall similarity rankings')



        return flask.render_template('visualize.html',cs_rec=cs_rec[:6],cs_rec_scores=cs_rec_scores[:6], 
                                                      ph_rec=ph_rec, ph_rec_scores=ph_rec_scores, 
                                                      pop_rec=pop_rec, pop_rec_scores=pop_rec_scores,
                                                      sim_rec=sim_rec, sim_rec_scores=sim_rec_scores,
                                                      user_code=user)

if __name__ == '__main__':
    app.run(debug=True)