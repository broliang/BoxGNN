import csv
from random import *
from collections import defaultdict
import pickle
from boxgraphv2 import RelGumbelBox
from BoxGraph import GumbelBox, BoxGraph
from conve import conve
from os.path import join
import torch
import dgl
import numpy as np

random_seed = 1


def get_vocab(filename):
    word2idx = defaultdict()
    with open(filename) as inputfile:
        lines = inputfile.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split('\t')
            word2idx[parts[1]] = parts[0]
    return word2idx


def get_new_model(params, g):
    if params.model == 'box':
        model = GumbelBox(params.device, params.VOCAB_SIZE, params.DIM, params.NEG_PER_POS,
                        [1e-4, 0.01], [-0.1, -0.001], params).to(params.device)
    elif params.model == 'box_':
        model = BoxGraph(params.device, params.VOCAB_SIZE, params.DIM, params.NEG_PER_POS,
                        [1e-4, 0.01], [-0.1, -0.001], params).to(params.device)
    elif params.model == 'rel':
        model = RelGumbelBox(params.device, params.VOCAB_SIZE, params.DIM, params.NEG_PER_POS,
                        [1e-4, 0.01], [-0.1, -0.001], params).to(params.device)
    elif params.model == 'conve':
        model = conve(params, g.edata['type'], g.edata['xxx'])
    model = model.to(params.device)
    return model


def load_hr_map(data_dir):
    file = join(data_dir, 'ndcg_test.pickle')
    with open(file, 'rb') as f:
        hr_map = pickle.load(f)
    return hr_map


def get_subset_of_given_relations(ids, rel_list):
    subs = []
    for r in rel_list:
        sub = ids[(ids[:, 1] == r).nonzero().squeeze(1)]  # sub triple set
        subs.append(sub)
    subset = torch.cat(subs, dim=0)
    return subset

def load_hr_map(data_dir):
    file = join(data_dir, 'ndcg_test.pickle')
    with open(file, 'rb') as f:
        hr_map = pickle.load(f)
    return hr_map


def build_graph(data, params):
    num_nodes = params.VOCAB_SIZE
    num_rels = params.REL_VOCAB_SIZE
    src = data.ids[:, 0]
    dst = data.ids[:, 2]
    edgesid = torch.tensor(data.ids[:, 1])
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src,dst)
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = in_deg ** -0.5
    norm[np.isinf(norm)] = 0
    norm = torch.tensor(norm)
    g.ndata['xxx'] = norm
    g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
    g.edata['type'] = edgesid
    return g

