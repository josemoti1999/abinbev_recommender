import pandas as pd

def get_item_user_details(item_code, user_code):
    item_details = pd.read_csv('preprocessed_data/item_details.csv').set_index('item')
    user_details = pd.read_csv('preprocessed_data/user_details.csv').set_index('user')
    if item_code in item_details.index:
        dict_1 = [['Brand',item_details.loc[item_code,'brand']],
        ['Subrand',item_details.loc[item_code,'subrand']],
        ['SEGMENTS : Pils / Spécialités / Superspécialités/Bouteille ',item_details.loc[item_code,'segments']],
        ['Container Type',item_details.loc[item_code,'cont_type']],
        ['Container Size',item_details.loc[item_code,'cont_size']],
        ['Variétés',item_details.loc[item_code,'varietes']],
        ['Segment LE',item_details.loc[item_code,'segment_le']],
        ['Degree Alcohol',item_details.loc[item_code,'degree_alc']],
        ]
    else:
        dict_1 = []
    if user_code in user_details.index:
        dict_2 = [['Groupement',user_details.loc[user_code,'groupement']],
        ['Postal Code',user_details.loc[user_code,'postal_code']],
        ['Street',user_details.loc[user_code,'street']],
        ['Sous groupement',user_details.loc[user_code,'Sous_groupement']],
        ['M2_Territory_ID',user_details.loc[user_code,'m2_id']],
        ['M1_Territory_ID',user_details.loc[user_code,'m1_id']],
        ['Dépt',user_details.loc[user_code,'dept']],
        ]
    else:
        dict_2 = []
    return dict_1, dict_2

def get_most_popular():
    popularity_df = pd.read_csv('preprocessed_data/popularity.csv').sort_values(by='user', ascending=False)
    popularity_df['scores'] = popularity_df['user']/(popularity_df['user'].max())
    pop_rec = list(popularity_df.loc[:,'item'].values)
    pop_rec_scores = list(popularity_df.loc[:,'scores'].values)
    return pop_rec, pop_rec_scores

def get_similar_items(ph_rec):
    item_details = pd.read_csv('preprocessed_data/item_details.csv').set_index('item')
    final = []
    for i,row in item_details.iterrows():
        feat2 = row.values.tolist()
        if i not in ph_rec:
            n = len(feat2)
            val = 0
            count=0
            for ele in ph_rec:
                ele = int(ele)
                if ele not in item_details.index:
                    continue
                count+=1
                feat1 = item_details.loc[ele,:].values.tolist()
                for a,b in zip(feat1,feat2):
                    val+=int(a==b)
            if count!=0:
                final.append((i,val/(n*count)))
    final.sort(key=lambda x:x[1], reverse=True)
    return final