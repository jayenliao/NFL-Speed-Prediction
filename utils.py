'''
TSA - NFL Project
Prdiction of Players' Speed
LSTM Experiments with Different Features

- Author: Jay C. Liao (jay-chiehen.liao@rennes-sb.com)
- Created on 4 Dec 2021
- Revised on 4 Dec 2021
- Finished on 5 Dec 2021
'''

import pickle, time, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def loading_and_preprocessing(dataPATH:str):

    # ---- (1) Data Loading ---- #
    print('Loading the original dataset ...')

    t0 = time.time()
    with open(dataPATH, 'rb') as f:
        tracking_small = pickle.load(f)
    print('Time cost of loading data: %8.2f s' % (time.time() - t0))

    # ---- (2) Data Preprocessing ---- #
    print('Preprocessing the dataset ...')

    for year, data in tracking_small.items():
        original_shape = data.shape
        no_missing = sum(pd.isnull(data['nflId']))
        data_new = data[-pd.isnull(data['nflId'])].copy()
        
        S1 = data_new['nflId'].copy()
        S2 = data_new['playId'].copy()
        S1 = S1.astype(str)
        S2 = S2.astype(str)
        S3 = S1 + S2
        data_new['tsId'] = S3

        no_unique_players = len(S1.unique())
        no_unique_plays = len(S2.unique())
        no_unique_TS = len(S3.unique())
        
        print(year, original_shape, no_missing, data_new.shape, no_unique_players, no_unique_plays, no_unique_TS)
        tracking_small[year] = data_new
    
    return tracking_small


def data2array(data:dict, features:list, target:str, save:bool):
    d_X, d_y, lst_playId = {}, {}, []

    for year in [2018, 2019, 2020]:
        d_X[year] = {}
        d_y[year] = {}
        loader = tqdm(data[year]['tsId'].unique())

        for tsId in loader:
            playId = int(tsId.split('.0')[-1])
            if playId not in lst_playId:
                lst_playId.append(playId)
                d_y[year][playId] = {}
                d_X[year][playId] = {}        
            condition = data[year]['tsId'] == tsId
            TS = data[year].loc[condition, target]

            for feature in features:
                if feature[:3] == 'lag':
                    lag = int(feature[-1])
                    data[year][feature] = 0
                    data[year].loc[condition, feature] = [0]*lag + list(TS[:-lag]) 
                    
            d_X[year][playId][tsId] = np.array(data[year][features][condition])
            d_y[year][playId][tsId] = np.array(TS)

    # Concatenate X arrays
    d_X_arr = {}
    n = len(features)
    for year, dict_year in d_X.items():
        d_X_arr[year] = {}
        for k, d in dict_year.items():
            d_X_arr[year][k] = np.concatenate([v.reshape(v.shape[0], n, 1) for v in d.values()], axis=2)

    if save:
        with open('./data/d_X_arr.pkl', 'wb') as f:
            pickle.dump(d_X_arr, f, pickle.HIGHEST_PROTOCOL)
        with open('./data/d_y.pkl', 'wb') as f:
            pickle.dump(d_y, f, pickle.HIGHEST_PROTOCOL)

    return d_X_arr, d_y

def arrays2tensors(d_X_arr:dict, d_y:dict, year:int, playId:int, train_size:float):
    assert train_size > 0 and train_size < 1

    array = d_X_arr[year][playId]
    seq_len = int(array.shape[0]*train_size)
    n_features = array.shape[1]
    n_observations = array.shape[-1]

    X_tr = torch.Tensor(array[:seq_len,:,:].reshape(n_observations, -1, n_features))
    X_te = torch.Tensor(array[seq_len:,:,:].reshape(n_observations, -1, n_features))
    
    y = np.concatenate([np.array(v).reshape(-1,1) for v in d_y[year][playId].values()], axis=1)
    y_tr = torch.tensor(y[:seq_len,:].reshape(n_observations, -1))
    y_te = torch.tensor(y[seq_len:,:].reshape(n_observations, -1))

    return X_tr, X_te, y_tr, y_te, seq_len


def plot_training_loss(loss:list, label:str, playId:int, year:int, savePATH:str):
    figTitle = label + "ing Loss of LSTM on The Prediction of Players' Speed in Play " + str(playId) + " in " + str(year)
    fn = savePATH + 'loss_plot_' + str(playId) + '_' + str(year) + '.png'
    plt.figure()
    plt.plot(loss, label=label)
    plt.title(figTitle)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(linestyle='--')
    plt.savefig(fn)
    print('The plot is saved as', fn)
