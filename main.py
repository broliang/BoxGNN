
from utils import get_new_model
import os
print(os.getcwd())
from utils import build_graph
from param import *
from trainer import run_train
import numpy as np
import torch
import random
import argparse
from dataset import *

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='main.py [<args>] [-h | --help]'
    )
    parser.add_argument('--data', type=str, default='cn15k', help="cn15k or nl27k")
    parser.add_argument('--task', type=str, default='mse', help="mse or ndcg")
    parser.add_argument('--gpu', type=int, help='which gpu')
    parser.add_argument('--model', type=str, default='two', help='which model')
    parser.add_argument('--reduce', type=str, default= 'mean', help='which reduce')
    parser.add_argument('--gnn', dest= 'gnn',action='store_true',
                        help='Use GNN?')
    parser.add_argument('--loss', type=str, default= 'kgc', help='kgc or else')

    return parser.parse_args(args)


def main(args):
    params = set_params(args.data, args.task, args.gpu, args.model, args.reduce, args.gnn, args.loss)


    # train_dataset = UncertainTrainTripleDataset(params.data_dir, 'train.tsv', params.REL_VOCAB_SIZE)
    # train_test_dataset = UncertainTrainTripleDataset(params.data_dir, 'train.tsv', params.REL_VOCAB_SIZE)
    # params.REL_VOCAB_SIZE = params.REL_VOCAB_SIZE * 2

    train_dataset = UncertainTripleDataset(params.data_dir, 'train.tsv')
    train_test_dataset = UncertainTripleDataset(params.data_dir, 'train.tsv')  # obsolete, not used

    dev_dataset = UncertainTripleDataset(params.data_dir, 'val.tsv')
    test_dataset = UncertainTripleDataset(params.data_dir, 'test.tsv')
    g= build_graph(train_dataset, params)
    g = g.to(params.device)
    print(params.whichmodel)
    print(params.early_stop)
    run = wandb_initialize(params)

    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)

    model = get_new_model(params, g)

    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.LR)

    run_train(g,
        model, run, train_dataset, train_test_dataset, dev_dataset, test_dataset,
        optimizer, params
    )

    print('done')


if __name__ =="__main__":
    main(parse_args())





