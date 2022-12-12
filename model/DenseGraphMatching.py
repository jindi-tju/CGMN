

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch_geometric.nn.dense.dense_gcn_conv import DenseGCNConv
from torch_geometric.nn.dense.dense_gin_conv import DenseGINConv
from torch_geometric.nn.dense.dense_sage_conv import DenseSAGEConv
from torch_geometric.utils import dense_to_sparse
from model.DenseGGNN import DenseGGNN
from sklearn.metrics import auc, roc_curve

from sklearn.utils import shuffle

import functools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn import metrics
#from tensorboardX import SummaryWriter

import numpy as np
from torch_geometric.utils import dropout_adj




class HierarchicalGraphMatchNetwork(torch.nn.Module):
    def __init__(self, node_init_dims, arguments, device):
        super(HierarchicalGraphMatchNetwork, self).__init__()

        self.node_init_dims = node_init_dims
        self.args = arguments
        self.device = device
        self.tau: float = 0.7


        self.dropout = arguments.dropout


        # ---------- Node Embedding Layer ----------
        filters = self.args.filters.split('_')
        self.gcn_filters = [int(n_filter) for n_filter in filters]  # GCNs' filter sizes
        self.gcn_numbers = len(self.gcn_filters)
        self.gcn_last_filter = self.gcn_filters[-1]  # last filter size of node embedding layer


        gcn_parameters = [
            dict(in_channels=self.gcn_filters[i - 1], out_channels=self.gcn_filters[i], bias=True) for i in
            range(1, self.gcn_numbers)
        ]
        gcn_parameters.insert(0, dict(in_channels=node_init_dims, out_channels=self.gcn_filters[0], bias=True))

        gin_parameters = [dict(nn=nn.Linear(in_features=self.gcn_filters[i - 1], out_features=self.gcn_filters[i])) for
                          i in range(1, self.gcn_numbers)]
        gin_parameters.insert(0, {'nn': nn.Linear(in_features=node_init_dims, out_features=self.gcn_filters[0])})

        ggnn_parameters = [dict(out_channels=self.gcn_filters[i]) for i in range(self.gcn_numbers)]

        conv_layer_constructor = {
            'gcn': dict(constructor=DenseGCNConv, kwargs=gcn_parameters),
            'graphsage': dict(constructor=DenseSAGEConv, kwargs=gcn_parameters),
            'gin': dict(constructor=DenseGINConv, kwargs=gin_parameters),
            'ggnn': dict(constructor=DenseGGNN, kwargs=ggnn_parameters)
        }

        conv = conv_layer_constructor[self.args.conv]
        constructor = conv['constructor']
        # build GCN layers
        setattr(self, 'gc{}'.format(1), constructor(**conv['kwargs'][0]))
        for i in range(1, self.gcn_numbers):
            setattr(self, 'gc{}'.format(i + 1), constructor(**conv['kwargs'][i]))


        setattr(self, 'gc_cross{}'.format(1), constructor(**conv['kwargs'][0]))
        for i in range(1, self.gcn_numbers):
            setattr(self, 'gc_cross{}'.format(i + 1), constructor(**conv['kwargs'][i]))

        self.fc1 = torch.nn.Linear(self.gcn_last_filter, self.gcn_last_filter)
        self.fc2 = torch.nn.Linear(self.gcn_last_filter, self.gcn_last_filter)



        self.global_fc_agg = nn.Linear(self.gcn_last_filter, self.gcn_last_filter)
        self.lstm_input_size = self.args.perspectives
        self.hidden_size = self.args.hidden_size
        self.agg_bilstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.hidden_size, num_layers=1,
                                  bidirectional=True, batch_first=True)
        self.agg_lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.hidden_size, num_layers=1,
                                  bidirectional=False, batch_first=True)

        # global aggregation
        self.global_flag = self.args.global_flag
        if self.global_flag is True:
            self.global_agg = self.args.global_agg
            if self.global_agg.lower() == 'max_pool':
                print("Only Max Pooling")
            elif self.global_agg.lower() == 'fc_max_pool':
                self.global_fc_agg = nn.Linear(self.gcn_last_filter, self.gcn_last_filter)
            elif self.global_agg.lower() == 'mean_pool':
                print("Only Mean Pooling")
            elif self.global_agg.lower() == 'fc_mean_pool':
                self.global_fc_agg = nn.Linear(self.gcn_last_filter, self.gcn_last_filter)
            elif self.global_agg.lower() == 'lstm':
                self.global_lstm_agg = nn.LSTM(input_size=self.gcn_last_filter, hidden_size=self.gcn_last_filter,
                                               num_layers=1, bidirectional=True, batch_first=True)
            else:
                raise NotImplementedError

        # ---------- Node-Graph Matching Layer ----------
        self.perspectives = self.args.perspectives  # number of perspectives for multi-perspective matching function
        if self.args.match.lower() == 'node-graph':
            self.mp_w = nn.Parameter(torch.rand(self.perspectives,
                                                self.gcn_last_filter))  # trainable weight matrix for multi-perspective matching function
            self.lstm_input_size = self.perspectives
        elif self.args.match.lower() == 'bilinear':
            self.bilinear = nn.Bilinear(in1_features=self.gcn_last_filter, in2_features=self.gcn_last_filter,
                                        out_features=self.perspectives)
            self.lstm_input_size = self.perspectives
        elif self.args.match.lower() == 'concat':
            self.nn = nn.Linear(in_features=2 * self.gcn_last_filter, out_features=self.perspectives)
            self.lstm_input_size = self.perspectives
        elif self.args.match.lower() == 'euccos':
            self.lstm_input_size = 2
        elif self.args.match.lower() == 'sub' or self.args.match.lower() == 'mul':
            self.lstm_input_size = self.gcn_last_filter
        elif self.args.match.lower() == 'submul':
            self.lstm_input_size = self.perspectives
            self.nn = nn.Linear(in_features=self.gcn_last_filter * 2, out_features=self.perspectives)

        # ---------- Aggregation Layer ----------
        self.hidden_size = self.args.hidden_size  # fixed the dimension size of aggregation hidden size
        # match aggregation
        if self.args.match_agg.lower() == 'bilstm':
            self.agg_bilstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.hidden_size, num_layers=1,
                                      bidirectional=True, batch_first=True)
        elif self.args.match_agg.lower() == 'lstm':
            self.agg_lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.hidden_size, num_layers=1,
                                    bidirectional=False, batch_first=True)
        elif self.args.match_agg.lower() == 'fc_avg' or self.args.match_agg.lower() == 'fc_max':
            self.fc_agg = nn.Linear(self.lstm_input_size, self.lstm_input_size)
        elif self.args.match_agg.lower() == 'avg' or self.args.match_agg.lower() == 'max':
            pass
        else:
            raise NotImplementedError

        # ---------- Prediction Layer ----------
        if self.args.task.lower() == 'regression':
            if self.global_flag is True:
                if self.global_agg.lower() == 'bilstm':
                    factor_global = 2
                else:
                    factor_global = 1
            else:
                factor_global = 0
            if self.args.match_agg == 'bilstm':
                factor_match_agg = 2
            else:
                factor_match_agg = 1
            factor = factor_match_agg + factor_global
            self.predict_fc1 = nn.Linear(int(self.hidden_size * 2), int(self.hidden_size))
            self.predict_fc2 = nn.Linear(int(self.hidden_size), int((self.hidden_size) / 2))
            self.predict_fc3 = nn.Linear(int((self.hidden_size) / 2), int((self.hidden_size) / 4))
            self.predict_fc4 = nn.Linear(int((self.hidden_size) / 4), 1)
        elif self.args.task.lower() == 'classification':
            print("classification task")
        else:
            raise NotImplementedError

    def global_aggregation_info(self, v, agg_func_name):
        """
        :param v: (batch, len, dim)
        :param agg_func_name:
        :return: (batch, len)
        """
        if agg_func_name.lower() == 'max_pool':
            agg_v = torch.max(v, 1)[0]
        elif agg_func_name.lower() == 'fc_max_pool':
            agg_v = self.global_fc_agg(v)
            agg_v = torch.max(agg_v, 1)[0]
        elif agg_func_name.lower() == 'mean_pool':
            agg_v = torch.mean(v, dim=1)
        elif agg_func_name.lower() == 'fc_mean_pool':
            agg_v = self.global_fc_agg(v)
            agg_v = torch.mean(agg_v, dim=1)
        elif agg_func_name.lower() == 'lstm':
            _, (agg_v_last, _) = self.global_lstm_agg(v)
            agg_v = agg_v_last.permute(1, 0, 2).contiguous().view(-1, self.gcn_last_filter * 2)
        else:
            raise NotImplementedError
        return agg_v

    @staticmethod
    def div_with_small_value(n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d



    def drop_feature(self, x, drop_prob):
        for i in range(x.size()[0]):


            drop_mask = torch.empty(
                (x[i].size(1),),
                dtype=torch.float32,
                device=x[i].device).uniform_(0, 1) < drop_prob
            x[i] = x[i].clone()
            x[i][:, drop_mask] = 0

        return x




    def cosine_attention(self, v1, v2):
        """
        :param v1: (batch, len1, dim)
        :param v2: (batch, len2, dim)
        :return:  (batch, len1, len2)
        """
        # (batch, len1, len2)
        a = torch.bmm(v1, v2.permute(0, 2, 1))

        v1_norm = v1.norm(p=2, dim=2, keepdim=True)  # (batch, len1, 1)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)  # (batch, len2, 1)
        d = v1_norm * v2_norm
        return self.div_with_small_value(a, d)

    def multi_perspective_match_func(self, v1, v2, w):
        """

        :param v1: (batch, len, dim)
        :param v2: (batch, len, dim)
        :param w: (perspectives, dim)
        :return: (batch, len, perspectives)
        """
        w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)  # (1,      1,  dim, perspectives)
        v1 = w * torch.stack([v1] * self.perspectives, dim=3)  # (batch, len, dim, perspectives)
        v2 = w * torch.stack([v2] * self.perspectives, dim=3)  # (batch, len, dim, perspectives)
        return functional.cosine_similarity(v1, v2, dim=2)  # (batch, len, perspectives)

    def forward_dense_gcn_layers(self, feat, adj):
        # TODO:
        feat_in = feat
        for i in range(1, self.gcn_numbers + 1):
            feat_out = functional.relu(getattr(self, 'gc{}'.format(i))(x=feat_in, adj=adj, mask=None, add_loop=False),
                                       inplace=True)
            feat_out = functional.dropout(feat_out, p=self.dropout, training=self.training)
            feat_in = feat_out
        return feat_out


    def forward_cross_gcn(self, feat, adj):
        # TODO:
        feat_in = feat
        for i in range(1, self.gcn_numbers + 1):
            feat_out = functional.relu(getattr(self, 'gc_cross{}'.format(i))(x=feat_in, adj=adj, mask=None, add_loop=False),
                                       inplace=True)
            feat_out = functional.dropout(feat_out, p=self.dropout, training=self.training)
            feat_in = feat_out
        return feat_out

    def forward(self, batch_x_p , batch_adj_p):


        feature_p1 = self.forward_dense_gcn_layers(feat=batch_x_p, adj = batch_adj_p)  # (batch, len_p, dim)

        return feature_p1


    def reg(self, agg_p, agg_h):

        x = torch.cat([agg_p, agg_h], dim=1)
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = functional.relu(self.predict_fc1(x))
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = functional.relu(self.predict_fc2(x))
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = functional.relu(self.predict_fc3(x))
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = self.predict_fc4(x)
        x = torch.sigmoid(x).squeeze(-1)
        return x

    def reshape(self, x, adj):
        B = x.size()[0]
        N = x.size()[1]
        D = x.size()[2]
        indices = []
        for i in range(B):
            edge_index = dense_to_sparse(adj[i])
            indices.append(edge_index[0] + i * N)
        edge_index = torch.cat(indices, dim=1)
        x = x.reshape(-1, D)
        return x, edge_index

    def projection(self, z: torch.Tensor) -> torch.Tensor:


        z = functional.elu(self.fc1(z))
        return self.fc2(z)


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):

        z1 = functional.normalize(z1)
        z2 = functional.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)

        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        ret = []

        lengh = z1.size()[0]
        for i in range(lengh):
            h1 = self.projection(z1[i])
            h2 = self.projection(z2[i])

            if batch_size == 0:
                l1 = self.semi_loss(h1, h2)
                l2 = self.semi_loss(h2, h1)
            else:
                l1 = self.batched_semi_loss(h1, h2, batch_size)
                l2 = self.batched_semi_loss(h2, h1, batch_size)
            ret_temp = (l1 + l2) * 0.5
            ret_temp = ret_temp.mean() if mean else ret_temp.sum()
            ret.append(ret_temp)

        return sum(ret)/len(ret)



    def aug_random_edge(self, input_adj, drop_percent):
        drop_percent = drop_percent / 2
        b = np.where(input_adj > 0,
                     np.random.choice(2, (input_adj.shape[0], input_adj.shape[0]), p=[drop_percent, 1 - drop_percent]),
                     input_adj)
        drop_num = len(input_adj.nonzero()[0]) - len(b.nonzero()[0])
        mask_p = drop_num / (input_adj.shape[0] * input_adj.shape[0] - len(b.nonzero()[0]))
        c = np.where(b == 0, np.random.choice(2, (input_adj.shape[0], input_adj.shape[0]), p=[1 - mask_p, mask_p]), b)

        return torch.from_numpy(c)



    def matching_layer_cross(self, feature_p, feature_h):
        # ---------- Node-Graph Matching Layer ----------

        attention = self.cosine_attention(feature_p, feature_h)  # (batch, len_p, len_h)

        attention_h = feature_h.unsqueeze(1) * attention.unsqueeze(
            3)  # (batch, 1, len_h, dim) * (batch, len_p, len_h, dim) => (batch, len_p, len_h, dim)
        attention_p = feature_p.unsqueeze(2) * attention.unsqueeze(
            3)  # (batch, len_p, 1, dim) * (batch, len_p, len_h, dim) => (batch, len_p, len_h, dim)

        att_mean_h = self.div_with_small_value(attention_h.sum(dim=2),
                                               attention.sum(dim=2, keepdim=True))  # (batch, len_p, dim)
        att_mean_p = self.div_with_small_value(attention_p.sum(dim=1),
                                               attention.sum(dim=1, keepdim=True).permute(0, 2,
                                                                                          1))  # (batch, len_h, dim)


        return att_mean_h, att_mean_p

    def matching_layer(self, feature_p, feature_h):
        # ---------- Node-Graph Matching Layer ----------

        attention = self.cosine_attention(feature_p, feature_h)  # (batch, len_p, len_h)

        attention_h = feature_h.unsqueeze(1) * attention.unsqueeze(
            3)  # (batch, 1, len_h, dim) * (batch, len_p, len_h, dim) => (batch, len_p, len_h, dim)
        attention_p = feature_p.unsqueeze(2) * attention.unsqueeze(
            3)  # (batch, len_p, 1, dim) * (batch, len_p, len_h, dim) => (batch, len_p, len_h, dim)

        att_mean_h = self.div_with_small_value(attention_h.sum(dim=2),
                                               attention.sum(dim=2, keepdim=True))  # (batch, len_p, dim)
        att_mean_p = self.div_with_small_value(attention_p.sum(dim=1),
                                               attention.sum(dim=1, keepdim=True).permute(0, 2,
                                                                                          1))  # (batch, len_h, dim)

        # ---------- Matching Layer ----------
        if self.args.match.lower() == "node-graph":
            multi_p = self.multi_perspective_match_func(v1=feature_p, v2=att_mean_h, w=self.mp_w)
            multi_h = self.multi_perspective_match_func(v1=feature_h, v2=att_mean_p, w=self.mp_w)
        elif self.args.match.lower() == 'bilinear':
            multi_p = self.bilinear(feature_p, att_mean_h)
            multi_h = self.bilinear(feature_h, att_mean_p)
        elif self.args.match.lower() == 'concat':
            multi_p = self.nn(torch.cat((feature_p, att_mean_h), dim=-1))
            multi_h = self.nn(torch.cat((feature_h, att_mean_p), dim=-1))
        elif self.args.match.lower() == 'euccos':
            batch_size = feature_p.shape[0]
            feature_p_reshape = feature_p.reshape(-1, self.gcn_last_filter)
            feature_h_reshape = feature_h.reshape(-1, self.gcn_last_filter)
            att_mean_p_reshape = att_mean_p.reshape(-1, self.gcn_last_filter)
            att_mean_h_reshape = att_mean_h.reshape(-1, self.gcn_last_filter)
            l2_dist_p = functional.pairwise_distance(feature_p_reshape, att_mean_h_reshape).reshape(
                batch_size, -1, 1)
            cosine_dist_p = functional.cosine_similarity(feature_p_reshape, att_mean_h_reshape).reshape(
                batch_size, -1, 1)
            l2_dist_h = functional.pairwise_distance(feature_h_reshape, att_mean_p_reshape).reshape(
                batch_size, -1, 1)
            cosine_dist_h = functional.cosine_similarity(feature_h_reshape, att_mean_p_reshape).reshape(
                batch_size, -1, 1)
            multi_p = torch.cat((l2_dist_p, cosine_dist_p), dim=-1)
            multi_h = torch.cat((l2_dist_h, cosine_dist_h), dim=-1)
        elif self.args.match.lower() == 'sub':
            multi_p = feature_p - att_mean_h
            multi_h = feature_h - att_mean_p
        elif self.args.match.lower() == 'mul':
            multi_p = feature_p * att_mean_h
            multi_h = feature_h * att_mean_p
        elif self.args.match.lower() == 'submul':
            multi_p = functional.relu(self.nn(torch.cat((feature_p - att_mean_h, feature_p * att_mean_h), dim=-1)))
            multi_h = functional.relu(self.nn(torch.cat((feature_h - att_mean_p, feature_h * att_mean_p), dim=-1)))
        else:
            raise NotImplementedError

        match_p = multi_p
        match_h = multi_h


        return  match_p, match_h


    def matching_layer_g1g2(self, feature_p, feature_h):
        # ---------- Node-Graph Matching Layer ----------

        attention = self.cosine_attention(feature_p, feature_h)  # (batch, len_p, len_h)

        attention_h = feature_h.unsqueeze(1) * attention.unsqueeze(
            3)  # (batch, 1, len_h, dim) * (batch, len_p, len_h, dim) => (batch, len_p, len_h, dim)
        attention_p = feature_p.unsqueeze(2) * attention.unsqueeze(
            3)  # (batch, len_p, 1, dim) * (batch, len_p, len_h, dim) => (batch, len_p, len_h, dim)

        att_mean_h = self.div_with_small_value(attention_h.sum(dim=2),
                                               attention.sum(dim=2, keepdim=True))  # (batch, len_p, dim)
        att_mean_p = self.div_with_small_value(attention_p.sum(dim=1),
                                               attention.sum(dim=1, keepdim=True).permute(0, 2,
                                                                                          1))  # (batch, len_h, dim)

        # ---------- euccos ---------
        batch_size = feature_p.shape[0]
        feature_p_reshape = feature_p.reshape(-1, self.gcn_last_filter)
        feature_h_reshape = feature_h.reshape(-1, self.gcn_last_filter)
        att_mean_p_reshape = att_mean_p.reshape(-1, self.gcn_last_filter)
        att_mean_h_reshape = att_mean_h.reshape(-1, self.gcn_last_filter)
        l2_dist_p = functional.pairwise_distance(feature_p_reshape, att_mean_h_reshape).reshape(
            batch_size, -1, 1)
        cosine_dist_p = functional.cosine_similarity(feature_p_reshape, att_mean_h_reshape).reshape(
            batch_size, -1, 1)
        l2_dist_h = functional.pairwise_distance(feature_h_reshape, att_mean_p_reshape).reshape(
            batch_size, -1, 1)
        cosine_dist_h = functional.cosine_similarity(feature_h_reshape, att_mean_p_reshape).reshape(
            batch_size, -1, 1)
        multi_p = torch.cat((l2_dist_p, cosine_dist_p), dim=-1)
        multi_h = torch.cat((l2_dist_h, cosine_dist_h), dim=-1)



        match_p = torch.cat((feature_p, multi_p),dim= -1)
        match_h = torch.cat((feature_h, multi_h),dim= -1)



        agg_p = torch.mean(match_p, dim=1)
        agg_h = torch.mean(match_h, dim=1)

        return agg_p, agg_h
        # return match_p, match_h

    def ginf(self, feature_p, feature_p0, global_agg):
        global_gcn_agg_p = self.global_aggregation_info(v=feature_p0, agg_func_name=global_agg)
        agg_p = torch.mean(feature_p, dim=1)
        feature_p = torch.cat([agg_p, global_gcn_agg_p], dim=1)

        return feature_p

    def GED_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, zp1: torch.Tensor, zp2: torch.Tensor,
                  n1: torch.Tensor, n2: torch.Tensor, np1: torch.Tensor, np2: torch.Tensor,
                  y_pos: torch.Tensor, y_neg: torch.Tensor,
                  ):
        f = lambda x: torch.exp(x / self.tau)

        z1 = torch.unsqueeze(z1, 0)
        z2 = torch.unsqueeze(z2, 0)
        zp1 = torch.unsqueeze(zp1, 0)
        zp2 = torch.unsqueeze(zp2, 0)
        # -------------graph-graph-pos--------------------
        between_sim = f(self.sim(z1, z2))
        between_sim_pos = f(self.sim(zp1, zp2))

        loss0 = -torch.log((between_sim_pos) / (between_sim + between_sim_pos))

        # -------regressionLoss----
        simneg = self.regression_loss(z1, z2)
        simpos = self.regression_loss(zp1, zp2)
        simneg = simneg.squeeze(-1)
        simpos = simpos.squeeze(-1)

        loss2 = torch.nn.functional.mse_loss(simneg, y_neg)
        loss3 = torch.nn.functional.mse_loss(simpos, y_pos)
        loss2 = (loss3 + loss2) * 0.5





        n1_n2 = z2.expand(len(n1), 100)
        n2_n1 = z1.expand(len(n2), 100)
        np1_np2 = zp2.expand(len(np1), 100)
        np2_np1 = zp1.expand(len(np2), 100)
        n2_np2 = zp2.expand(len(n2), 100)
        np2_n2 = z2.expand(len(np2), 100)

        # -------------node-graph-pos--------------------
        between_sim_pos_node1 = f(self.sim(np1, np1_np2))
        between_sim_pos_node2 = f(self.sim(np2, np2_np1))
        # -------------node-graph-neg--------------------
        between_sim_neg_node1 = f(self.sim(n1, n1_n2))
        between_sim_neg_node2 = f(self.sim(n2, n2_n1))
        between_sim_neg_node3 = f(self.sim(np2, np2_n2))
        between_sim_neg_node4 = f(self.sim(n2, n2_np2))
        # ----------------------------------------------
        x1 = torch.mean(between_sim_pos_node1.sum(1))
        x2 = torch.mean(between_sim_pos_node2.sum(1))
        x3 = torch.mean(between_sim_neg_node1.sum(1))
        x4 = torch.mean(between_sim_neg_node2.sum(1))
        x5 = torch.mean(between_sim_neg_node3.sum(1))
        x6 = torch.mean(between_sim_neg_node4.sum(1))

        loss1 = -torch.log((x1 + x2)
                           / (x1 + x2 + x3 + x4 + x5 + x6))


        return loss2 * self.mse_loss + loss1 * self.node_graph_loss + loss0 * self.graph_graph_loss, simpos, simneg

    def GED_loss(self, z1: torch.Tensor, z2: torch.Tensor,zp1: torch.Tensor, zp2: torch.Tensor,y_pos:torch.Tensor, y_neg:torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        ret = []
        n1 = z1
        n2 = z2
        np1 = zp1
        np2 = zp2


        z1 = torch.mean(z1, dim=1)
        z2 = torch.mean(z2, dim=1)
        zp1 = torch.mean(zp1, dim=1)
        zp2 = torch.mean(zp2, dim=1)

        predict_pos = []
        predict_neg = []



        lengh = z1.size()[0]
        for i in range(lengh):
            h1 = self.projection(z1[i])
            h2 = self.projection(z2[i])
            hp1 = self.projection(zp1[i])
            hp2 = self.projection(zp2[i])

            nh1 = self.projection(n1[i])
            nh2 = self.projection(n2[i])
            nhp1 = self.projection(np1[i])
            nhp2 = self.projection(np2[i])


            if batch_size == 0:
                l1,simpos1, simneg1 = self.GED_semi_loss(h1, h2, hp1, hp2, nh1,nh2,nhp1,nhp2, y_pos[i],y_neg[i])
                l2,simpos2, simneg2 = self.GED_semi_loss(h2, h1, hp2, hp1, nh2,nh1,nhp2,nhp1, y_pos[i],y_neg[i])
                predict_pos.append((simpos2+simpos1)*0.5)
                predict_neg.append((simneg2+simneg1)*0.5)
            else:
                l1 = self.batched_semi_loss(h1, h2, batch_size)
                l2 = self.batched_semi_loss(h2, h1, batch_size)
            ret_temp = (l1 + l2) * 0.5
            ret_temp = ret_temp.mean() if mean else ret_temp.sum()
            ret.append(ret_temp)



        return sum(ret) / len(ret),predict_pos,predict_neg

    def regression(self, z1: torch.Tensor, z2: torch.Tensor):
        # -------regressionLoss----
        x = torch.cat([z1, z2], dim=1)
        x = functional.dropout(x, p=self.dropout, training=self.training)
        # x = self.predict_fc1(x)
        x = functional.relu(self.predict_fc1(x))
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = functional.relu(self.predict_fc2(x))
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = functional.relu(self.predict_fc3(x))
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = self.predict_fc4(x)
        x = torch.sigmoid(x).squeeze(-1)
        return x
    def regression1(self, z1: torch.Tensor):

        x=z1
        x = functional.dropout(x, p=self.dropout, training=self.training)

        x = functional.relu(self.predict_fc1(x))
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = functional.relu(self.predict_fc2(x))
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = functional.relu(self.predict_fc3(x))
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = self.predict_fc4(x)
        x = torch.sigmoid(x).squeeze(-1)
        return x

    def prediction(self, feature_p, feature_h,y):

        if self.args.match_agg.lower() == 'bilstm':
            _, (agg_p_last, _) = self.agg_bilstm(feature_p)  # (batch, seq_len, l) -> (2, batch, hidden_size)
            agg_p = agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2)

            _, (agg_h_last, _) = self.agg_bilstm(feature_h)
            agg_h = agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2)
        elif self.args.match_agg.lower() == 'lstm':
            _, (agg_p_last, _) = self.agg_lstm(feature_p)
            agg_p = agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size)

            _, (agg_h_last, _) = self.agg_lstm(feature_h)
            agg_h = agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size)
        elif self.args.match_agg.lower() == 'avg':
            agg_p = torch.mean(feature_p, dim=1)
            agg_h = torch.mean(feature_h, dim=1)
        elif self.args.match_agg.lower() == 'fc_avg':
            agg_p = torch.mean(self.fc_agg(feature_p), dim=1)
            agg_h = torch.mean(self.fc_agg(feature_h), dim=1)
        elif self.args.match_agg.lower() == 'max':
            agg_p = torch.max(feature_p, dim=1)[0]
            agg_h = torch.max(feature_h, dim=1)[0]
        elif self.args.match_agg.lower() == 'fc_max':
            agg_p = torch.max(self.fc_agg(feature_p), dim=1)[0]
            agg_h = torch.max(self.fc_agg(feature_h), dim=1)[0]
        else:
            raise NotImplementedError

        # option: global aggregation
        if self.global_flag is True:
            global_gcn_agg_p = self.global_aggregation_info(v=feature_p, agg_func_name=self.global_agg)
            global_gcn_agg_h = self.global_aggregation_info(v=feature_h, agg_func_name=self.global_agg)

            agg_p = torch.cat([agg_p, global_gcn_agg_p], dim=1)
            agg_h = torch.cat([agg_h, global_gcn_agg_h], dim=1)



        if self.args.task.lower() == 'regression':
            x = torch.cat([agg_p, agg_h], dim=1)
            x = functional.dropout(x, p=self.dropout, training=self.training)
            x = functional.relu(self.predict_fc1(x))
            x = functional.dropout(x, p=self.dropout, training=self.training)
            x = functional.relu(self.predict_fc2(x))
            x = functional.dropout(x, p=self.dropout, training=self.training)
            x = functional.relu(self.predict_fc3(x))
            x = functional.dropout(x, p=self.dropout, training=self.training)
            x = self.predict_fc4(x)
            x = torch.sigmoid(x).squeeze(-1)
            return x
        elif self.args.task.lower() == 'classification':

            x = torch.cat([agg_p, agg_h], dim=1)
            # y = torch.cat([y,y],dim=0)
            X = x.detach().cpu().numpy()
            Y = y.detach().cpu().numpy()
            Y = Y.reshape(-1, 1)

            X, Y= shuffle(X, Y)

            X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                                test_size=0.3)

            logreg = LogisticRegression(solver='liblinear', C=1.0, penalty='l2')


            logreg.fit(X_train, y_train)

            y_pred = logreg.predict(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_pred)
            model_auc = auc(fpr, tpr)

            return model_auc
        else:
            raise NotImplementedError


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        # self.early_stop = False
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.counter = 0
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        if self.verbose:
            print(f'Validation score decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        torch.save(model, 'finish_model.pkl')
        self.val_loss_min = val_loss