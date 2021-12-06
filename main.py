'''
TSA - NFL Project
Prdiction of Players' Speed
LSTM Experiments with Different Features

- Author: Jay C. Liao (jay-chiehen.liao@rennes-sb.com)
- Created on 4 Dec 2021
- Revised on 4 Dec 2021
- Finished on 5 Dec 2021
'''

import os, pickle, torch
import numpy as np
import pandas as pd
from datetime import datetime
from args import init_arguments
from utils import loading_and_preprocessing, data2array, arrays2tensors, plot_training_loss
from model import myLSTM, train
from torch.utils.data import TensorDataset, DataLoader

def main(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    savePATH = args.savePATH if args.savePATH[-1] == '/' else args.savePATH + '/'
    savePATH += datetime.strftime(datetime.now(), '%y-%m-%d_%H-%M-%S') + '/'
    try:
        os.mkdir(savePATH)
    except:
        pass

    # (1) Loading and preprocess the data

    try:
        with open('./data/d_X_arr.pkl', 'rb') as f:
            d_X_arr = pickle.load(f)
        with open('./data/d_y.pkl', 'rb') as f:
            d_y = pickle.load(f)
    except:
        tracking_small = loading_and_preprocessing(args.dataPATH)
        d_X_arr, d_y = data2array(tracking_small, args.features, args.target, True)


    # (2) Set and train models

    dict_tensor = {'X_tr': {}, 'X_te': {}, 'y_tr': {}, 'y_te': {}}
    dict_models = {}
    dict_loss_tr = {}
    dict_loss_te = {}

    n_features = len(args.features)
    for year in args.years:
        dict_models[year] = {}
        dict_loss_tr[year] = {}
        dict_loss_te[year] = {}

        for playId in d_X_arr[year].keys():
            X_tr, X_te, y_tr, y_te, seq_len = arrays2tensors(d_X_arr, d_y, year, playId, args.train_size)
            #dict_tensor = {'X_tr': {}, 'X_te': {}, 'y_tr': {}, 'y_te': {}}    
            dataset_tr = TensorDataset(X_tr.to(device), y_tr.to(device))
            dataset_te = TensorDataset(X_te.to(device), y_te.to(device))
            loader_tr = DataLoader(dataset_tr, batch_size=args.batch_size)
            loader_te = DataLoader(dataset_te, batch_size=args.batch_size)

            dict_models[playId] = myLSTM(n_features, args.hidden_size, args.n_layers, seq_len, args.batch_size, device)
            dict_models[playId] = dict_models[playId].to(device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(dict_models[playId].parameters(), lr=args.learning_rate) 
            loss_tr = train(playId, dict_models[playId], args.epochs, loader_tr, optimizer, criterion, device)
            dict_loss_tr[year][playId] = loss_tr
            fn = savePATH + 'loss_tr_' + str(playId) + '_' + str(year) + '.txt'
            np.savetxt(fn, np.array(loss_tr))
            plot_training_loss(loss_tr, 'Train', playId, year, savePATH)
            #dict_loss_te[year][playId] = test(dict_models[playId], loader_te, criterion, device)
    
    with open(savePATH + 'dict_loss_tr.pkl', 'wb') as f:
        pickle.dump(dict_loss_tr, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = init_arguments().parse_args()
    main(args)