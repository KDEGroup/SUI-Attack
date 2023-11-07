import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from collections import defaultdict

def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = torch.sum(mx, 0)
        r_inv = torch.pow(rowsum, -0.5)
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        colsum = torch.sum(mx, 1)
        c_inv = torch.pow(colsum, -0.5)
        c_inv[torch.isinf(c_inv)] = 0.
        c_mat_inv = torch.diag(c_inv)
        mx = torch.matmul(mx, r_mat_inv.int())
        mx = torch.matmul(c_mat_inv.int(), mx)
        return mx


class Encoder(Module):

    def __init__(self, input_dim, hidden_dim, num_users, num_items, num_classes, act, dropout, bias=True):
        super(Encoder, self).__init__()

        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.u_weight = Parameter(torch.randn(num_classes, input_dim, hidden_dim))
        self.v_weight = Parameter(torch.randn(num_classes, input_dim, hidden_dim))
        if bias:
            self.u_bias = Parameter(torch.randn(hidden_dim))
            self.v_bias = self.u_bias
        else:
            self.u_bias = None
            self.v_bias = None

        for w in [self.u_weight, self.v_weight]:
            nn.init.xavier_normal_(w)

    def forward(self, u_feat, v_feat, u, v, support):

        u_feat = self.dropout(u_feat)
        v_feat = self.dropout(v_feat)

        supports_u = []
        supports_v = []
        u_weight, v_weight = 0, 0
        for r in range(support.shape[0]):
            u_weight = u_weight + self.u_weight[r]
            v_weight = v_weight + self.v_weight[r]

            tmp_u = torch.mm(u_feat, u_weight)
            tmp_v = torch.mm(v_feat, v_weight)

            support_norm = normalize(support[r])
            support_norm_t = normalize(support[r].t())
        
            supports_u.append(torch.mm(support_norm[u], tmp_v.int()))
            supports_v.append(torch.mm(support_norm_t[v], tmp_u.int()))
        z_u = torch.sum(torch.stack(supports_u, 0), 0)
        z_v = torch.sum(torch.stack(supports_v, 0), 0)
        if self.u_bias is not None:
            z_u = z_u + self.u_bias
            z_v = z_v + self.v_bias

        u_outputs = self.act(z_u)
        v_outputs = self.act(z_v)
        return u_outputs, v_outputs


class Decoder(Module):
    def __init__(self, num_users, num_items, input_dim, output_dim, init,
                 dropout=0.7):
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, output_dim)
        )
        self.init = nn.init.xavier_normal_
        # self.init = select(init)
        self.weight.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            self.init(m.weight)
            if m.bias is not None:
                self.init(m.bias)

    def forward(self, u_hidden, v_hidden):

        u_hidden = self.dropout(u_hidden)
        v_hidden = self.dropout(v_hidden)

        u_w = self.weight(u_hidden)
        v_w = self.weight(v_hidden)

        return u_w, v_w

class GAE(nn.Module):
    def __init__(self, num_users, num_items, num_classes,
                       u_features, v_features,
                       hidden, dropout, args):
        super(GAE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.dropout = dropout

        self.u_features = u_features
        self.v_features = v_features

        input_dim = u_features.shape[1]

        self.projector = nn.Sequential(
            nn.Linear(input_dim, 32, bias=False),
            nn.LeakyReLU(),
            nn.Linear(32, hidden[0], bias=False)
        )

        self.gcl1 = Encoder(hidden[0], hidden[1],
                                    num_users, num_items,
                                    num_classes, torch.relu, self.dropout, bias=True)
        self.gcl2 = Encoder(hidden[1], hidden[2],
                                    num_users, num_items,
                                    num_classes, torch.relu, self.dropout, bias=True)

        self.denseu = nn.Linear(hidden[2], hidden[3], bias=False)
        self.densev = nn.Linear(hidden[2], hidden[3], bias=False)

        self.decoder = Decoder(num_users=num_users, num_items=num_items,
                                         output_dim=input_dim,
                                         input_dim=hidden[3],
                                         dropout=0.5, init=args.init)
        self.inf_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.load_influence_pretrain(args.dataName, args.recommender)
        self.load_encoder_from_pretrained(args.dataName, args.recommender)

    def load_influence_pretrain(self, dataset, model_name):
        self.inf_predictor.load_state_dict(torch.load(f'./saved/influence/{dataset}_{model_name}.pt'))
        for param in self.inf_predictor.parameters():
            param.requires_grad = False

    def load_encoder_from_pretrained(self, dataset, model_name):
        self.gcl1.load_state_dict(torch.load(f'./saved/pretrain/{dataset}_{model_name}_1.pt'))
        self.gcl2.load_state_dict(torch.load(f'./saved/pretrain/{dataset}_{model_name}_2.pt'))

    def forward(self, u, v, r_matrix, mask, do_train=True):
        if do_train:
            u_features = self.u_features
            u_features[mask] = 0.
        else:
            attack_size = mask.shape[0]
            fake_emb = torch.zeros((attack_size, self.feat_dim))
            u_features = torch.cat((self.user_feat, fake_emb), dim=0)

        u_features = self.projector(self.u_features)
        v_features = self.projector(self.v_features)

        u_z, v_z = self.gcl1(u_features, v_features, torch.arange(self.num_users), torch.arange(self.num_items), r_matrix)
        u_z, v_z = self.gcl2(u_z, v_z, u, v, r_matrix)
        u_h = self.denseu(F.dropout(u_z, self.dropout))
        v_h = self.densev(F.dropout(v_z, self.dropout))

        recon_u, recon_v = self.decoder(u_h, v_h)
        score = self.inf_predictor(recon_u[mask])
    
        return recon_u, recon_v, score

def gumbel_top_k(logits, k, temperature=1.0, eps=1e-10):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    
    gumbel_logits = (logits + gumbel_noise) / temperature
    
    _, top_k_indices = torch.topk(gumbel_logits, k, dim=-1, largest=True, sorted=False)
    
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(-1, top_k_indices, 1)
    
    return one_hot

class SubGraphGen(nn.Module):
    def __init__(self, user_feat, item_feat, num_classes, trainMatrix, args):
        super(SubGraphGen).__init__()

        drop, pop_thr, hidden = args.drop, args.pop_thr, args.hidden
        self.feat_dim, self.trainMatrix = user_feat.shape[1], trainMatrix

        self.user_feat, self.item_feat  = user_feat, item_feat
        self.hot_items = self.popular(trainMatrix, pop_thr)
        self.num_users, self.num_items = user_feat.shape[0], item_feat.shape[0]   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.feat_gen = GAE(self.num_users, self.num_items, num_classes, user_feat, item_feat, hidden, drop, args)
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim,  self.feat_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feat_dim * 2, self.feat_dim)
        )
        self.activate = nn.LeakyReLU()
        self.get_adj_list(trainMatrix)
    
    def get_adj_list(self, trainMatrix):
        user_adj_list, item_adj_list = defaultdict(set), defaultdict(set)
        for user in range(trainMatrix.shape[0]):
            items = np.nonzero(trainMatrix[user, :])[0]
            user_adj_list[user].update(items)
        for item in range(trainMatrix.shape[1]):
            users = np.nonzero(trainMatrix[:, item])[0]
            item_adj_list[item].update(users)
        self.user_adj_list = user_adj_list
        self.item_adj_list = item_adj_list  

    def popular(self, trainMatrix, pop_thr):
        item_popularity = np.count_nonzero(trainMatrix, axis=0)
        hot_items = np.argsort(item_popularity)[::-1][:int(pop_thr * trainMatrix.shape[1])]
        return hot_items
    
    def fusion(self, target, attack_size, u, v, r_matrix, do_train):
        mask = random.sample(u, attack_size)
        recon_u, recon_v, score = self.feat_gen(u, v, r_matrix, mask, do_train=do_train)
        fake_user_emb = recon_u[mask]
        if do_train:
            user_emb = self.projector(fake_user_emb)
            return user_emb, recon_u, recon_v, score
        else:
            sechop = set()
            for neighbor in list(self.item_adj_list[target]):
                neighbors_item = self.user_adj_list[neighbor]
                indices = [i + self.n_user for i in neighbors_item]
                sechop.update(indices)
            
            return user_emb, sechop, recon_u, recon_v, score
        
    def stack(self, tensor):
        sub_tensors = []
        for i in range(self.num_classes):
            mask = (tensor == i)
            sub_tensors.append(mask.int() * i)
        output_tensor = (torch.stack(sub_tensors, dim=0))
        return output_tensor

    def forward(self, target, attack_size, budget, n_ran, do_train):
        r_matrix = self.stack(self.trainMatrix)
        candiate = None
        if do_train:
            fake_emb, recon_u, recon_v, score = self.fusion(target, attack_size, range(self.num_users), range(self.num_items), r_matrix, do_train)
        else:
            fake_emb, candidate, recon_u, recon_v, score = self.fusion(target, attack_size, range(self.num_users), range(self.num_items), r_matrix, do_train)
        node_emb = self.projector(self.item_feat)
        edges = F.cosine_similarity(fake_emb.unsqueeze(1), node_emb.unsqueeze(0), dim=-1)
        edges = gumbel_top_k(edges, budget)
        
        if not do_train:
            candiate = np.tile(np.array(list(candidate)), (attack_size, 1))
            random_pop = np.array([np.random.choice(self.hot_items, n_ran) for _ in range(attack_size)])
            candiate = np.concatenate((candiate, random_pop), axis=1)
            node_emb = self.projector(self.item_feat)[candiate]
        return recon_u, recon_v, score, edges, candidate
    



