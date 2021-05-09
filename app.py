import flask
from flask import render_template, url_for, redirect, session
import os
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from cs_train import train, predict

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

        session['details'] = defaultdict(list)
        for i,ele in enumerate(cs_rec):
            session['details'][int(ele)].append(f'1. This product has never been bought before.')
            session['details'][int(ele)].append(f'2. Predicted rating score of {cs_rec_scores[i]} and is {i}th ranked among cross-selling opportunities product recommended')
        
        for i,ele in enumerate(ph_rec):
            session['details'][int(ele)].append(f'3. This product has been bought multiple times before')
            session['details'][int(ele)].append(f'4. Has a rating score of {ph_rec_scores[i]} according to our rating algorithm and is {i}th ranked among the items bought till now')
            
        return flask.render_template('visualize.html',cs_rec=cs_rec[:6],cs_rec_scores=cs_rec_scores[:6], 
        ph_rec=ph_rec, ph_rec_scores=ph_rec_scores, user_code=user)
        #return flask.render_template('positive.html',cs_rec=cs_rec,cs_rec_scores=cs_rec_scores,user_code=user)

if __name__ == '__main__':
    app.run(debug=True)