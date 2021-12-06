'''
TSA - NFL Project
Prdiction of Players' Speed
LSTM Experiments with Different Features

- Author: Jay C. Liao (jay-chiehen.liao@rennes-sb.com)
- Created on 4 Dec 2021
- Revised on 4 Dec 2021
- Finished on 5 Dec 2021
'''

import argparse

def init_arguments():
    parser = argparse.ArgumentParser(prog="TSA - Predict players' speed by using LSTM")

    # General
    parser.add_argument('--random_state', type=int, default=4028)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dataPATH', '-dP', type=str, default='./data/', help='The path where should the data be loaded in.')
    parser.add_argument('--savePATH', type=str, default='./output/', help='The path to store the outputs, including models, plots, and training and evalution results.')
    parser.add_argument('--features', nargs='+', type=str, default=['x', 'y', 'o', 'lag1', 'lag2'])
    parser.add_argument('--years', nargs='+', type=int, default=[2018, 2019, 2020])
    parser.add_argument('--target', type=str, default='s')

    # Model structure
    parser.add_argument('--n_layers', type=int, default=1, help='No. of convolution layers')
    parser.add_argument('--hidden_size', type=int, default=2, help='Dimension of the hidden state of the linear layers')
    parser.add_argument('--hidden_act', type=str, default='ReLU', choices=['Sigmoid', 'ReLU', 'tanh'], help='Activation function of all hidden layers (except for the last layer)')
    
    # Training
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda', 'cuda:1'], help='Device name')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Momentum', 'AdaGrad', 'Adam'], help='Optimizer')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=300, help='No. of epochs')
    parser.add_argument('--batch_size', '-bs', type=int, default=11, help='Batch size')
    parser.add_argument('--train_size', type=float, default=.7)
    
    return parser