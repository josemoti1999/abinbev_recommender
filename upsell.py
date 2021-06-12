import pandas as pd

def get_similar_items(ph_rec):
    item_details_all = pd.read_csv('preprocessed_data/item_details.csv').set_index('item')
    deg_alc = item_details_all.pop('degree_alc')
    item_details = item_details_all
    final_dict = {}

    for ele in ph_rec:
        final = []
        for i,row in item_details.iterrows():
            feat2 = row.values.tolist()
            if i not in ph_rec:
                n = len(feat2)
                val = 0
                count=0
                ele = int(ele)
                if ele not in item_details.index:
                    continue
                count+=1
                feat1 = item_details.loc[ele,:].values.tolist()
                for a,b in zip(feat1,feat2):
                    val+=int(a==b)
                val+= (1-(abs(deg_alc.loc[ele]-deg_alc.loc[i]))/9)
                if count!=0:
                    final.append([i,val/((n+1)*count)])
        final.sort(key=lambda x: x[1], reverse=True)
        final_dict[ele] = final

    return final_dict

def get_average_similarity(ph_rec):
    item_details_all = pd.read_csv('preprocessed_data/item_details.csv').set_index('item')
    deg_alc = item_details_all.pop('degree_alc')
    item_details = item_details_all
    final = []
    avg_smt={}
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
                val+= (1-(abs(deg_alc.loc[ele]-deg_alc.loc[i]))/9)
            if count!=0:
                final.append([i, val/((n+1)*count)])
    for x,y in final:
        avg_smt[x]=y
    return avg_smt

def get_final_scores(final_dict, avg_smt):
    final_scores = []
    for k in final_dict:
        for c in final_dict[k]:
            if c[0] in avg_smt.keys():
                ini_score = c[1]
                fin_score = ini_score*avg_smt[c[0]]
                final_scores.append([k, c[0], fin_score])

    final_scores.sort(key=lambda x:x[2], reverse=True)
    return final_scores

def upsell_recommendation(ph_rec):
    cost_df = pd.read_csv('preprocessed_data/cost.csv').set_index('Material')
    final_dict = get_similar_items(ph_rec)
    avg_smt = get_average_similarity(ph_rec)
    final_scores = get_final_scores(final_dict, avg_smt)
    recom_items = []
    source_item = []
    recom_score = []
    recom_costs = []
    source_cost = []
    cnt=0
    for i in final_scores:
        if i[1] not in recom_items and cost_df.loc[i[1],'MACO/HL ']>cost_df.loc[i[0],'MACO/HL '] and cnt<10:
            recom_items.append(i[1])
            source_item.append(i[0])
            recom_score.append(i[2])
            recom_costs.append(cost_df.loc[i[1],'MACO/HL '])
            source_cost.append(cost_df.loc[i[0],'MACO/HL '])
            cnt+=1
    return recom_items, source_item, recom_score, recom_costs, source_cost