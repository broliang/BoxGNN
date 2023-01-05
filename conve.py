import torch
from torch import nn
import dgl
from layer import CompGCNCov
import torch.nn.functional as F


class CompGCN(nn.Module):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 conv_bias=True, gcn_drop=0., opn='mult'):
        super(CompGCN, self).__init__()
        self.act = torch.tanh
        self.loss = nn.BCELoss()
        self.num_ent, self.num_rel, self.num_base = num_ent, num_rel, num_base
        self.init_dim, self.gcn_dim, self.embed_dim = init_dim, gcn_dim, embed_dim
        self.conv_bias = conv_bias
        self.gcn_drop = gcn_drop
        self.opn = opn
        self.edge_type = edge_type  # [E]
        self.edge_norm = edge_norm  # [E]
        self.n_layer = n_layer

        self.init_embed = self.get_param([self.num_ent, self.init_dim])  # initial embedding for entities
        if self.num_base > 0:
            # linear combination of a set of basis vectors
            self.init_rel = self.get_param([self.num_base, self.init_dim])
        else:
            # independently defining an embedding for each relation
            self.init_rel = self.get_param([self.num_rel * 2, self.init_dim])

        self.conv1 = CompGCNCov(self.init_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=self.num_base,
                                num_rel=self.num_rel)
        self.conv2 = CompGCNCov(self.gcn_dim, self.embed_dim, self.act, conv_bias, gcn_drop,
                                opn) if n_layer == 2 else None
        self.bias = nn.Parameter(torch.zeros(self.num_ent))

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def forward_base(self, g, subj, rel, obj, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed, self.init_rel  # embedding of relations
        x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
        x = drop1(x)  # embeddings of entities [num_ent, dim]
        x, r = self.conv2(g, x, r, self.edge_type, self.edge_norm) if self.n_layer == 2 else (x, r)
        x = drop2(x) if self.n_layer == 2 else x
        sub_emb = torch.index_select(x, 0, subj)  # filter out embeddings of subjects in this batch
        rel_emb = torch.index_select(r, 0, rel)  # filter out embeddings of relations in this batch
        obj_emb = torch.index_select(x, 0, obj)  # filter out embeddings of subjects in this batch

        return sub_emb, rel_emb, obj_emb


class conve(CompGCN):
    """
    :param num_ent: number of entities
    :param num_rel: number of different relations
    :param num_base: number of bases to use
    :param init_dim: initial dimension
    :param gcn_dim: dimension after first layer
    :param embed_dim: dimension after second layer
    :param n_layer: number of layer
    :param edge_type: relation type of each edge, [E]
    :param bias: weather to add bias
    :param gcn_drop: dropout rate in ARGATcov
    :para: combination operator
    :param hid_drop: gcn output (embedding of each entity) dropout
    :param input_drop: dropout in conve input
    :param conve_hid_drop: dropout in conve hidden layer
    :param feat_drop: feature dropout in conve
    :param num_filt: number of filters in conv2d
    :param ker_sz: kernel size in conv2d
    :param k_h: height of 2D reshape
    :param k_w: width of 2D reshape
    """
    def __init__(self, params, edge_type, edge_norm, bias= True, hid_drop=0.2, input_drop=0.2, conve_hid_drop=0.2, feat_drop=0.2,
                 num_filt=200, ker_sz=7, k_h=20, k_w=10, ratio = 1):
        super(conve, self).__init__(params.VOCAB_SIZE, params.REL_VOCAB_SIZE, -1, 100, 100, 200, 2,
                                            edge_type, edge_norm, True, 0.3, 'corr')
        self.num_ent, self.num_rel,  self.device = params.VOCAB_SIZE, params.REL_VOCAB_SIZE, params.device
        self.init_dim, self.gcn_dim, self.embed_dim = 100, 100, 200
        self.ratio = ratio
        self.best_t = 0
        self.params = params
        self.drop = nn.Dropout(hid_drop)
        self.loss = nn.BCEWithLogitsLoss()
        self.loss_margin = nn.SoftMarginLoss()
        self.hid_drop, self.input_drop, self.conve_hid_drop, self.feat_drop = hid_drop, input_drop, conve_hid_drop, feat_drop

        self.num_filt = num_filt
        self.ker_sz, self.k_w, self.k_h = ker_sz, k_w, k_h

        self.bn0 = torch.nn.BatchNorm2d(1)  # one channel, do bn on initial embedding
        self.bn1 = torch.nn.BatchNorm2d(self.num_filt)  # do bn on output of conv
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        self.drop = torch.nn.Dropout(self.hid_drop)  # gcn output dropout
        self.input_drop = torch.nn.Dropout(self.input_drop)  # stacked input dropout
        self.feature_drop = torch.nn.Dropout(self.feat_drop)  # feature map dropout
        self.hidden_drop = torch.nn.Dropout(self.conve_hid_drop)  # hidden layer dropout

        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_filt,
                                      kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=bias)

        flat_sz_h = int(2 * self.k_h) - self.ker_sz + 1  # height after conv
        flat_sz_w = self.k_w - self.ker_sz + 1  # width after conv
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)  # fully connected projection

        self.ent_embed = nn.Embedding(self.num_ent, self.init_dim)
        self.rel_embed = nn.Embedding(self.num_rel, self.init_dim)
        self.score_func = nn.Linear(self.init_dim, 1)
    def calc_loss(self, pred, label):
        # return self.loss_margin(pred,label)
        return self.loss(pred, label)
    def concat(self, ent_embed, rel_embed):
        """
        :param ent_embed: [batch_size, embed_dim]
        :param rel_embed: [batch_size, embed_dim]
        :return: stack_input: [B, C, H, W]
        """
        ent_embed = ent_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_input = torch.cat([ent_embed, rel_embed], 1)  # [batch_size, 2, embed_dim]
        assert self.embed_dim == self.k_h * self.k_w
        stack_input = stack_input.reshape(-1, 1, 2 * self.k_h, self.k_w)  # reshape to 2D [batch, 1, 2*k_h, k_w]
        return stack_input
    def forward(self, g, ids, probs, train=True):
        head, rel, tail = ids[:, 0], ids[:, 1], ids[:, 2]

        head_emb, rel_emb, tail_emb = self.forward_base(g, head, rel, tail, self.drop, self.input_drop)
        rel_embed = self.rel_embed(rel)
        stack_input = self.concat(head_emb, rel_embed)
        x = self.bn0(stack_input)
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x * tail_emb
        score = self.score_func(x)
        return score.squeeze(), probs

    def random_negative_sampling(self, positive_samples, pos_probs, neg_per_pos=None):
        if neg_per_pos is None:
            neg_per_pos = self.ratio
        negative_samples1 = torch.repeat_interleave(positive_samples, neg_per_pos, dim=0)
        negative_samples2 = torch.repeat_interleave(positive_samples, neg_per_pos, dim=0)

        corrupted_heads = [self.get_negative_samples_for_one_positive(pos, neg_per_pos, mode='corrupt_head') for pos
                           in positive_samples]
        corrupted_tails = [self.get_negative_samples_for_one_positive(pos, neg_per_pos, mode='corrupt_tail') for pos
                           in positive_samples]

        negative_samples1[:, 0] = torch.cat(corrupted_heads)
        negative_samples2[:, 2] = torch.cat(corrupted_tails)
        negative_samples = torch.cat((negative_samples1, negative_samples2), 0).to(self.device)
        neg_probs = torch.zeros(negative_samples.shape[0], dtype=pos_probs.dtype).to(self.device)

        return negative_samples, neg_probs

    def get_negative_samples_for_one_positive(self, positive_sample, neg_per_pos, mode):
        head, relation, tail = positive_sample
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < neg_per_pos:
            negative_sample = np.random.randint(self.params.VOCAB_SIZE, size=neg_per_pos * 2)

            # filter true values
            if mode == 'corrupt_head' and (int(relation), int(tail)) in self.true_head:  # filter true heads
                # For test data, some (relation, tail) pairs may be unseen and not in self.true_head
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(int(relation), int(tail))],
                    assume_unique=True,
                    invert=True
                )
                negative_sample = negative_sample[mask]
            elif mode == 'corrupt_tail' and (int(head), int(relation)) in self.true_tail:
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(int(head), int(relation))],
                    assume_unique=True,
                    invert=True
                )
                negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:neg_per_pos]

        negative_sample = torch.from_numpy(negative_sample)
        return negative_sample