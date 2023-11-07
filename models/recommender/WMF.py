import torch
import os, math
import torch.nn as nn

from .metrics import *
from .recommender import Recommender
import torch.nn.functional as F

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

def _array2sparsediag(x):
    values = x
    indices = np.vstack([np.arange(x.size), np.arange(x.size)])

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = [x.size, x.size]

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def save_checkpoint(path, model, optimizer=None, epoch=-1):
    """Save model checkpoint and optimizer state to file"""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None
    }
    file_path = "%s.pt" % path
    print("Saving checkpoint to {}".format(file_path))
    torch.save(state, file_path)

class WeightedMF(nn.Module):
    def __init__(self, n_users, n_items, hidden_dim):
        super(WeightedMF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.dim = hidden_dim  # WMF can only have one latent dimension.

        self.Q = nn.Parameter(torch.zeros([self.n_items, self.dim]).normal_(mean=0, std=0.1))
        self.P = nn.Parameter(torch.randn([self.n_users, self.dim]).normal_(mean=0, std=0.1))
        self.params = nn.ParameterList([self.Q, self.P])

    def forward(self, user_id=None, item_id=None):
        if user_id is None and item_id is None:
            return torch.mm(self.P, self.Q.t())
        if user_id is not None:
            return torch.mm(self.P[[user_id]], self.Q.t())
        if item_id is not None:
            return torch.mm(self.P, self.Q[[item_id]].t())

    def get_norm(self, user_id=None, item_id=None):
        l2_reg = torch.norm(self.P[[user_id]], p=2, dim=-1).sum() + torch.norm(self.Q[[item_id]], p=2, dim=-1).sum()
        return l2_reg



class Trainer(Recommender):
    def __init__(self, optim_method='sgd', dim=64, num_epoch=50, weight_alpha=None, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

        self.dim = dim

        if optim_method == 'sgd':
            self.num_epoch = num_epoch
            self.save_feq = num_epoch
        elif optim_method == 'als':
            self.num_epoch = 20
            self.save_feq = 20
        self.batch_size = 128
        self.lr = 0.005
        self.l2 = 1e-5

        self.weight_alpha = 20
        self.optim_method = optim_method
        self.prepare_data()
        self.build_network()

    def build_network(self):
        self.net = WeightedMF(self.n_users, self.n_items, self.dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.l2)

    def train(self):
        if self.optim_method not in ('sgd', 'als'):
            raise ValueError("Unknown optim_method {} for WMF.".format(self.optim_method))

        if self.optim_method == 'sgd':
            self.train_sgd()
        if self.optim_method == 'als':
            self.train_als()

    def train_als(self):
        best_perf = 0.0
        for epoch in range(1, self.num_epoch + 1):
            data = self.train_matrix

            model = self.net
            P = model.P.detach()
            Q = model.Q.detach()

            weight_alpha = self.weight_alpha - 1
            # Using Pytorch for ALS optimization
            # Update P
            lambda_eye = torch.eye(self.dim).to(self.device) * self.l2
            # residual = Q^tQ + lambda*I
            residual = torch.mm(Q.t(), Q) + lambda_eye
            for user, batch_data in enumerate(data):
                # x_u: N * 1
                x_u = sparse2tensor(batch_data).to(self.device).t()
                cu = batch_data.toarray().squeeze() * weight_alpha + 1
                Cu = _array2sparsediag(cu).to(self.device)
                Cu_minusI = _array2sparsediag(cu - 1).to(self.device)
                # Q^tCuQ + lambda*I = Q^tQ + lambda*I + Q^t(Cu-I)Q
                # left hand side
                lhs = torch.mm(Q.t(), Cu_minusI.mm(Q)) + residual
                # right hand side
                rhs = torch.mm(Q.t(), Cu.mm(x_u))

                new_p_u = torch.mm(lhs.inverse(), rhs)
                model.P.data[user] = new_p_u.t()

            # Update Q
            data = data.transpose()
            # residual = P^tP + lambda*I
            residual = torch.mm(P.t(), P) + lambda_eye
            for item, batch_data in enumerate(data):
                # x_v: M x 1
                x_v = sparse2tensor(batch_data).to(self.device).t()
                # Cv = diagMat(alpha * rating + 1)
                cv = batch_data.toarray().squeeze() * weight_alpha + 1
                Cv = _array2sparsediag(cv).to(self.device)
                Cv_minusI = _array2sparsediag(cv - 1).to(self.device)
                # left hand side
                lhs = torch.mm(P.t(), Cv_minusI.mm(P)) + residual
                # right hand side
                rhs = torch.mm(P.t(), Cv.mm(x_v))

                new_q_v = torch.mm(lhs.inverse(), rhs)
                model.Q.data[item] = new_q_v.t()

            # print("[TRIAN recommender WMF_als], epoch: {}".format(epoch))

            if epoch % self.save_feq == 0:
                result = self.evaluate(verbose=False)
                # Save model checkpoint if it has better performance.
                if result[self.golden_metric] > best_perf:
                    str_metric = "{}={:.4f}".format(self.golden_metric, result[self.golden_metric])
                    # print("Having better model checkpoint with performance {}(epoch:{})".format(str_metric, epoch))
                    self.path_save = os.path.join(self.dir_model, 'WMF_als')
                    save_checkpoint(self.path_save, self.net, self.optimizer, epoch)
                    best_perf = result[self.golden_metric]

    def train_sgd(self):
        best_perf = 0.0
        for epoch in range(1, self.num_epoch + 1):
            data = self.train_matrix

            n_rows = data.shape[0]
            idx_list = np.arange(n_rows)

            # Set model to training mode.
            model = self.net.to(self.device)
            model.train()
            np.random.shuffle(idx_list)

            epoch_loss = 0.0
            for batch_idx in minibatch(idx_list, batch_size=self.batch_size):
                batch_tensor = sparse2tensor(data[batch_idx]).to(self.device)
                # Compute loss
                outputs = model(user_id=batch_idx)
                # l2_norm = model.get_norm(user_id=batch_idx)

                loss = mse_loss(data=batch_tensor,
                                logits=outputs,
                                weight=self.weight_alpha).sum()
                # loss += l2_norm * 10
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % self.save_feq == 0:
                result = self.evaluate(verbose=False)
                # Save model checkpoint if it has better performance.
                if result[self.golden_metric] > best_perf:
                    str_metric = "{}={:.4f}".format(self.golden_metric, result[self.golden_metric])
                    # print("Having better model checkpoint with performance {}(epoch:{})".format(str_metric, epoch))
                    self.path_save = os.path.join(self.dir_model, 'WMF_sgd')
                    save_checkpoint(self.path_save, self.net, self.optimizer, epoch)
                    best_perf = result[self.golden_metric]

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        # Set model to eval mode.
        model = self.net.to(self.device)
        model.eval()

        n_rows = data.shape[0]
        idx_list = np.arange(n_rows)
        recommendations = np.empty([n_rows, top_k], dtype=np.int64)

        # Make predictions first, and then sort for top-k.
        with torch.no_grad():
            data_tensor = sparse2tensor(data).to(self.device)
            preds = model()

        if return_preds:
            all_preds = preds
        if not allow_repeat:
            preds[data.nonzero()] = -np.inf
        if top_k > 0:
            _, recs = preds.topk(k=top_k, dim=1)
            recommendations = recs.cpu().numpy()
        if return_preds:
            return recommendations, all_preds.cpu()
        else:
            return recommendations

    def evaluate_attack(self, ratings, trainMatrix, topk=20):
        if isinstance(ratings, torch.Tensor):
            ratings = ratings.detach().cpu().numpy()
        if isinstance(trainMatrix, torch.Tensor):
            trainMatrix = trainMatrix.detach().cpu().numpy()

        mask = trainMatrix != 0
        ratings[mask] = -np.inf
        hc, nc = 0, 0
        for i in range(self.n_users):
            idx = np.argsort(ratings[i])[::-1][:topk]
            hc += self.target in idx
            nc += math.log(2) / math.log(np.where(idx == self.target)[0] + 2) if self.target in idx else 0
        hr = hc / self.n_users
        ndcg = nc / self.n_users
        return hr, ndcg

    def fit_withPQ(self, P, Q):
        # get self.train_matrix(real+fake) and self.test_matrix
        self.prepare_data()

        self.build_network()
        self.net.P.data = torch.tensor(P)
        self.net.Q.data = torch.tensor(Q)

        result1 = self.evaluate()

        result2 = self.validate()
        return result1, result2

    def fit_adv(self, fake_tensor, target_items, ratio):
        import higher

        data_tensor = sparse2tensor(self.train_matrix)
        data_tensor = data_tensor.to(self.device)
        data_tensor = torch.concat([data_tensor, fake_tensor], dim=0).to(self.device)
        data_tensor.requires_grad_()
        
        # target_tensor = torch.ones((self.n_users, 1))
        target_tensor = torch.zeros_like(data_tensor)
        target_tensor[:, target_items] = 1.0

        model = self.net.to(self.device)
        model.train()
        for _ in range(1, self.num_epoch + 1):
            n_rows = data_tensor.shape[0]
            idx_list = np.arange(n_rows)
            np.random.shuffle(idx_list)
            for batch_idx in minibatch(idx_list, batch_size=self.batch_size):
                batch_tensor = data_tensor[batch_idx].to(self.device)
                outputs = model(user_id=batch_idx)
                loss = mse_loss(data=batch_tensor,
                                logits=outputs,
                                weight=self.weight_alpha).sum()
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
        with higher.innerloop_ctx(model, self.optimizer) as (fmodel, diffopt):
            for i in range(int(self.num_epoch * ratio)):
                np.random.shuffle(idx_list)
                epoch_loss = 0.0
                fmodel.train()
                for batch_idx in minibatch(
                        idx_list, batch_size=self.batch_size):
                    batch_tensor = data_tensor[batch_idx]
                    # Compute loss
                    outputs = fmodel(batch_idx)
                    loss = mse_loss(data=batch_tensor,
                                    logits=outputs,
                                weight=self.weight_alpha).sum()
                    epoch_loss += loss.item()
                    diffopt.step(loss)
        fmodel.eval()
        predictions = fmodel()
        n_fakes = fake_tensor.shape[0]
        adv_loss = mult_ce_loss(
            logits=predictions[-n_fakes:, ],
            data=target_tensor[-n_fakes:, ]
        ).sum()
        model.load_state_dict(fmodel.state_dict())
        return adv_loss
                