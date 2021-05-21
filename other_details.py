import pandas as pd

def get_item_details(item_code):
    item_details = pd.read_csv('preprocessed_data/item_details.csv').set_index('item')
    if item_code in item_details.index:
        dict_ = [['Brand',item_details.loc[item_code,'brand']],
        ['Subrand',item_details.loc[item_code,'subrand']],
        ['SEGMENTS : Pils / Spécialités / Superspécialités/Bouteille ',item_details.loc[item_code,'segments']],
        ['Container Type',item_details.loc[item_code,'cont_type']],
        ['Variétés',item_details.loc[item_code,'cont_size']],
        ['Segment LE',item_details.loc[item_code,'segment_le']]]
        return dict_
    else:
        return []

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