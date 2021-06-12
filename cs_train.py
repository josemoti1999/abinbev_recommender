from cross_sell import *

def model(input_shape, q_dims, p_dims):
    #Encoder 
    model_input=tf.keras.Input(input_shape)
    layer = tf.keras.layers.Dense(q_dims[1],activation='tanh')(model_input)
    mean=tf.keras.layers.Dense(q_dims[-1])(layer)
    var=tf.keras.layers.Dense(q_dims[-1])(layer)
    encoder_model=tf.keras.models.Model(model_input,[mean,var])
    decoder_input=tf.keras.Input((q_dims[-1],))
    layer = tf.keras.layers.Dense(p_dims[1],activation='tanh')(decoder_input)
    layer = tf.keras.layers.Dense(p_dims[-1], activation='sigmoid')(layer)
    decoder_model=tf.keras.models.Model(decoder_input,layer)  
    mean,var=encoder_model(model_input)
    epsilon=tf.random.normal(shape=(tf.shape(var)[0],tf.shape(var)[1]))
    z=mean+tf.exp(var)*epsilon
    model_out=decoder_model(z)
    model=tf.keras.models.Model(model_input,model_out)
    
    #kl_loss = - 0.5 * K.sum(1 + var - K.square(mean) - K.exp(var), axis=-1)
    return model,decoder_model,encoder_model

def step(model_input, model,opt):
    # a step in training loop using custom tf functions
    mask = tf.cast(tf.where(model_input==-1,0,1),dtype=tf.float32)
    with tf.GradientTape() as tape:
        model_out = model(model_input)
        mse = tf.keras.losses.mse(mask*model_input,mask*model_out)
        loss = tf.reduce_mean(mse)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return model, model_out, loss

def do_validation(pred, val_te):
    # evaluates mse and rndcg of trained models
    # mse decreases from 0.25 to 0.06 and rndcg increases from 0.5 to 0.89
    # rndcg is a modified version of ndcg where the rankning of recommendations is also taken into account
    try:
        pred = pred.numpy()
    except:
        pass
    val_mse = 0
    ndcg = 0
    for a, grouped in val_te.groupby('user'):
        actual, predicted = [],[]
        for i, row in grouped.iterrows():
            u, i = int(row['user']), int(row['item'])
            a = row['rating']
            val_mse += (pred[u][i]-a)**2
            actual.append(a)
            predicted.append(pred[u][i])
        actual = np.array(actual)
        predicted = np.array(predicted)

        temp = np.argsort(actual)[::-1]
        actual = np.empty_like(temp)
        actual[temp] = np.arange(1,len(actual)+1)

        temp = np.argsort(predicted)[::-1]
        predicted = np.empty_like(temp)
        predicted[temp] = np.arange(1,len(predicted)+1)

        arg_actual = np.argsort(actual)[::-1]
        actual_sorted = np.take(actual,arg_actual)
        pred_sorted = np.take(predicted,arg_actual)
        discounts = np.log2(np.arange(len(actual)) + 2)
        dcg = np.sum(pred_sorted/discounts)
        dcg_ = np.sum(pred_sorted[::-1]/discounts)
        idcg = np.sum(actual_sorted/discounts)
        idcg_ = np.sum(actual_sorted[::-1]/discounts)
        ndcg += (dcg*idcg_)/(dcg_*idcg)
    return val_mse/val_te.shape[0], ndcg/len(val_te['user'].unique())

def personalization_index(matrix, ui):
    # 1 corresponds to a good personalization to everyone
    # 0 corresponds to same recommendations for everyone
    # variational encoders have a low personalization compared to nmf
    ui_mask = np.where(ui==-1,1,0)
    K = 10
    scores_list = []
    for u in range(matrix.shape[0]):
        u = int(u)
        scores = matrix[u,:]
        cs_scores = ui_mask[u,:]*scores
        cs_rec = np.argpartition(cs_scores,-K)[-K:]
        cs_rec = cs_rec[np.argsort(cs_scores[cs_rec])][::-1]
        cs_rec_scores = cs_scores[cs_rec]
        scores_list.append(cs_rec)
    scores_list = np.stack(scores_list, axis=0)
    a = set(scores_list[0,:])
    pi = 0
    for u in range(1,matrix.shape[0]):
        b = set(scores_list[u,:])
        pi += len(a.intersection(b))/10
    return 1-(pi/(matrix.shape[0]-1))

def train(retrain=False):
    # training based on variational autoencoders
    # the best model is saved which is then retreived while predicting
    # for nmf the model fitting is done while predicting
    start = time.time()

    if not os.path.exists('preprocessed_data/train_min_max.csv') or retrain==True:
        print('Preprocesssing')
        ui, train_min_max = preprocess(retrain)
    else:
        ui, train_min_max = np.load('preprocessed_data/ui.npy'), pd.read_csv('preprocessed_data/train_min_max.csv')
    n_items = ui.shape[-1]
    p_dims = [40,80,n_items]
    q_dims = p_dims[::-1]

    train_full = pd.read_csv('preprocessed_data/train_full.csv')
    val_te = pd.read_csv('preprocessed_data/val_te.csv')
    input_shape = (ui.shape[1],)
    epochs = 200
    opt = tf.keras.optimizers.Adam(lr=0.001)
    vae_model,_,_=model(input_shape, q_dims, p_dims)

    best_rndcg = float('-inf')
    for i in range(epochs):
        vae_model, pred, mse = step(ui,vae_model,opt)
        val_mse, rndcg_val = do_validation(pred, val_te)
        if rndcg_val>best_rndcg:
            minval = val_mse
            best_loss = mse
            best_rndcg = rndcg_val
            best_epoch = i    
            vae_model.save_weights('preprocessed_data/best_weights/')
    final_model,_,_=model(input_shape,q_dims,p_dims)
    final_model.load_weights('preprocessed_data/best_weights/')
    pred_final = final_model.predict(ui)
    a,b = do_validation(pred_final,val_te)
    print('For VAE based model:')
    print('MSE-->',a, ', RNDCG-->',b)
    print('PI-->',personalization_index(pred_final,ui))

    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(train_full[['user', 'item', 'rating']], reader)
    data_test = Dataset.load_from_df(val_te[['user', 'item', 'rating']], reader)
    trainset = data.build_full_trainset()
    testset = data_test.build_full_trainset()
    testset = testset.build_testset()
    algo = SVD(n_factors = 10, n_epochs=10)
    predictions = algo.fit(trainset).test(testset)
    #accuracy.mse(predictions)
    ui_nmf_pred = np.zeros((178,176))
    for u in range(178):
        for i in range(176):
            ui_nmf_pred[u][i] = algo.predict(u,i)[3]
    np.save('preprocessed_data/ui_nmf_pred.npy',ui_nmf_pred)
    a,b = do_validation(ui_nmf_pred,val_te)
    print('For NMF based model:')
    print('MSE-->',a, ', RNDCG-->',b)
    print('PI-->',personalization_index(ui_nmf_pred,ui))

    final__ = 0.1*pred_final + 0.9*ui_nmf_pred
    a,b = do_validation(final__, val_te)
    print('For combined model')
    print('MSE-->',a, ', RNDCG-->',b)
    print('PI-->', personalization_index(final__,ui))
    print('Time to complete-->',time.time()-start)


def predict(u):
    # 90% of score from nmf and 10% score from variational autoencoders
    # using variational encoders doesnt give a boost in rmse and ndcg because
    # given dataset can be simply represented using linear nmf models
    u = int(u)
    ui = np.load('preprocessed_data/ui.npy')
    n_items = ui.shape[-1]
    p_dims = [40,80,n_items]
    q_dims = p_dims[::-1]
    val_te = pd.read_csv('preprocessed_data/val_te.csv')
    input_shape = (ui.shape[1],)
    ui_full = ui[u,:].copy()
    for _,row in val_te.iterrows():
        ui_full[int(row['item'])] = row['rating']
    ui_mask = np.where(ui_full==-1,1,0)
    ui_ph = np.where(ui_full==-1,0,1)
    final_model,_,_=model(input_shape,q_dims,p_dims)
    final_model.load_weights('preprocessed_data/best_weights/')
    pred_final = final_model.predict(ui)


    ui_nmf_pred = np.load('preprocessed_data/ui_nmf_pred.npy')
    
    final__ = 0.1*pred_final + 0.9*ui_nmf_pred
    scores = final__[u,:]
    cs_scores = ui_mask*scores
    temp = cs_scores[cs_scores>0]
    cs_rec = np.argsort(cs_scores)[::-1][:temp.shape[0]]
    cs_rec_scores = cs_scores[cs_rec]
    ph_scores = ui_ph*ui_full
    temp = ph_scores[ph_scores>0]
    ph_rec = np.argsort(ph_scores)[::-1][:temp.shape[0]]
    ph_rec_scores = ph_scores[ph_rec]
    return cs_rec, cs_rec_scores, ph_rec, ph_rec_scores
