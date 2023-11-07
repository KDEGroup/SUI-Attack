import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .metrics import *
import torch.nn.functional as F
from .recommender import Recommender

def minibatch(*tensors, **kwargs):
    """Mini-batch generator for pytorch tensor."""
    batch_size = kwargs.get('batch_size', 128)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

EPSILON = 1e-12

def mult_ce_loss(data, logits):
    """Multi-class cross-entropy loss."""
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -log_probs * data

    instance_data = data.sum(1)
    instance_loss = loss.sum(1)
    # Avoid divide by zeros.
    res = instance_loss / (instance_data + EPSILON)
    return res

def mse_loss(data, logits, weight):
    """Mean square error loss."""
    weights = torch.ones_like(data)
    weights[data > 0] = weight
    res = weights * (data - logits) ** 2
    return res.sum(1)

def sparse2tensor(sparse_data):
    """Convert sparse csr matrix to pytorch tensor."""
    return torch.FloatTensor(sparse_data.toarray())

class ItemAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ItemAE, self).__init__()
        self.q_dims = [input_dim] + hidden_dim
        self.p_dims = self.q_dims[::-1]

        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(self.q_dims[:-1], self.q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(self.p_dims[:-1], self.p_dims[1:])])


    def encode(self, input):
        h = input
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            h = torch.tanh(h)
        return h

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def forward(self, input):
        z = self.encode(input)
        return self.decode(z)

    def loss(self, data, outputs):
        return mse_loss(data, outputs, weight=20)


class Trainer(Recommender):
    def __init__(self, hidden_dim=[256, 128], num_epoch=50, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = hidden_dim

        self.lr = 1e-3
        self.l2 = 1e-6
        self.num_epoch=50
        self.batch_size = 128
        self.valid_batch_size = 64
        self.prepare_data()
        self.build_network()

    def build_network(self):
        self.net = ItemAE(
            self.n_users, hidden_dim=self.dim).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.l2)
    def train(self):
        for _ in range(self.num_epoch):
            self.train_epoch(self.train_matrix)
    def train_epoch(self, data):
        # Transpose the data first for ItemVAE.
        data = data.transpose()

        n_rows = data.shape[0]
        n_cols = data.shape[1]
        idx_list = np.arange(n_rows)

        # Set model to training mode.
        model = self.net.to(self.device)
        model.train()
        np.random.shuffle(idx_list)

        epoch_loss = 0.0
        batch_size = (self.batch_size
                      if self.batch_size > 0 else len(idx_list))
        for batch_idx in minibatch(
                idx_list, batch_size=batch_size):
            batch_tensor = sparse2tensor(data[batch_idx]).to(self.device)

            # Compute loss
            outputs = model(batch_tensor)
            loss = model.loss(data=batch_tensor,
                              outputs=outputs).sum()
            epoch_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return epoch_loss

    def fit_adv(self, fake_tensor, target_items, ratio):
        import higher

        data_tensor = sparse2tensor(self.train_matrix)
        data_tensor = torch.concat([data_tensor, fake_tensor], dim=0).to(self.device)
        data_tensor.requires_grad_()

        target_tensor = torch.zeros_like(data_tensor)
        target_tensor[:, target_items] = 1.0
        data_tensor = data_tensor.t()

        n_rows = data_tensor.shape[0]
        n_cols = data_tensor.shape[1]
        idx_list = np.arange(n_rows)

        # Set model to training mode.
        model = self.net.to(self.device)
        optimizer = self.optimizer

        batch_size = (self.batch_size
                      if self.batch_size > 0 else len(idx_list))
        for i in range(1, self.num_epoch + 1):
            t1 = time.time()
            np.random.shuffle(idx_list)
            model.train()
            epoch_loss = 0.0
            for batch_idx in minibatch(
                    idx_list, batch_size=batch_size):
                batch_tensor = data_tensor[batch_idx]
                # Compute loss

                outputs = model(batch_tensor)
                loss = model.loss(data=batch_tensor,
                                  outputs=outputs).sum()
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
            print("Switching to higher mode...")
            for i in range(int(self.num_epoch * ratio)):
                t1 = time.time()
                np.random.shuffle(idx_list)
                epoch_loss = 0.0
                fmodel.train()
                for batch_idx in minibatch(
                        idx_list, batch_size=batch_size):
                    batch_tensor = data_tensor[batch_idx]
                    # Compute loss
                    outputs = fmodel(batch_tensor)
                    loss = fmodel.loss(data=batch_tensor,
                                       outputs=outputs).sum()
                    epoch_loss += loss.item()
                    diffopt.step(loss)


            fmodel.eval()
            all_preds = list()
            for batch_idx in minibatch(np.arange(n_rows),
                                       batch_size=batch_size):
                all_preds += [fmodel(data_tensor[batch_idx])]
            predictions = torch.cat(all_preds, dim=0).t()

            # Compute adversarial (outer) loss.
            n_fakes = fake_tensor.shape[0]
            adv_loss = mult_ce_loss(
                logits=predictions[-n_fakes:, ],
                data=target_tensor[-n_fakes:, ]).sum()
            # Copy fmodel's parameters to default trainer.net().
            model.load_state_dict(fmodel.state_dict())

        return adv_loss

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        # Set model to eval mode
        model = self.net.to(self.device)
        model.eval()

        # Transpose the data first for ItemVAE.
        data = data.transpose()

        n_rows = data.shape[0]
        n_cols = data.shape[1]

        idx_list = np.arange(n_rows)
        recommendations = np.empty([n_cols, top_k], dtype=np.int64)

        # Make predictions first, and then sort for top-k.
        all_preds = list()
        with torch.no_grad():
            for batch_idx in minibatch(
                    idx_list, batch_size=self.valid_batch_size):
                data_tensor = sparse2tensor(data[batch_idx]).to(self.device)
                preds = model(data_tensor)
                all_preds.append(preds)

        all_preds = torch.cat(all_preds, dim=0).t()
        data = data.transpose()
        idx_list = np.arange(n_cols)
        for batch_idx in minibatch(
                idx_list, batch_size=self.valid_batch_size):
            batch_data = data[batch_idx].toarray()
            preds = all_preds[batch_idx]
            if not allow_repeat:
                preds[batch_data.nonzero()] = -np.inf
            if top_k > 0:
                _, recs = preds.topk(k=top_k, dim=1)
                recommendations[batch_idx] = recs.cpu().numpy()

        if return_preds:
            return recommendations, all_preds.cpu()
        else:
            return recommendations