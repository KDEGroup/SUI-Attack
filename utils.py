import numpy as np
import torch, os
import matplotlib.pyplot as plt
import scipy.sparse as sp
import random
import pandas as pd
import pickle as pkl

# check the parameters and gradients
def show_param(model):
    for name, param in model.named_parameters():
        print(name, ': ', param)
    
def get_param_dict(model):
    param_dict = dict(list(model.named_parameters()))
    return param_dict

def show_grad(optimizer):
    print(
        [x.grad for x in optimizer.param_groups[0]['params']]
    )

def check_print(words):
    print(words)
    exit(-1)

def move(vars, device):
    for i, v in enumerate(vars):
        vars[i] = v.to(device)
    return tuple(vars)
# ==========================================================

def load_fake_array(file_path, rate=None):
    fake_array = np.load(file_path, allow_pickle=True)
    fake_array[fake_array > 0] = 1 if rate is not None else rate
    return sp.csr_matrix(fake_array)


def save_checkpoint(model, optimizer, path, epoch=-1):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    file_path = "%s.pt" % path
    torch.save(state, file_path)

def load_checkpoint(path):
    file_path = "%s.pt" % path
    state = torch.load(file_path, map_location=torch.device('cpu'))
    return state["epoch"], state["state_dict"], state["optimizer"]


def show_train_loss(losses, save=False):
    plt.plot(losses)
    plt.ylabel('loss of each iteration')
    plt.xlabel('iteration')
    if save:
        plt.save('just_loss.jpg')
    plt.show()


def normalize_features(feat):
    degree = np.asarray(feat.sum(1)).flatten()
    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    if feat_norm.nnz == 0:
        print('ERROR: normalized adjacency matrix has only zero entries!!!!!')
        exit

    return feat_norm

def preprocess_user_item_features(u_features, v_features):

    zero_csr_u = sp.csr_matrix((u_features.shape[0], v_features.shape[1]), dtype=u_features.dtype)
    zero_csr_v = sp.csr_matrix((v_features.shape[0], u_features.shape[1]), dtype=v_features.dtype)

    # num_u, u_dim+i_dim
    u_features = sp.hstack([u_features, zero_csr_u], format='csr')
    v_features = sp.hstack([zero_csr_v, v_features], format='csr')

    return u_features, v_features


def load_data(fname, seed=1234, verbose=True):

    def map_data(data):
        uniq = list(set(data))

        id_dict = {old: new for new, old in enumerate(sorted(uniq))}
        data = np.array(list(map(lambda x: id_dict[x], data)))
        n = len(uniq)

        return data, id_dict, n

    data_dir = 'data/' + fname
    files = ['/u.data', '/u.item', '/u.user']
    sep = '\t'
    filename = data_dir + files[0]

    dtypes = {
        'u_nodes': np.int32, 'v_nodes': np.int32,
        'ratings': np.float32, 'timestamp': np.float64}

    data = pd.read_csv(
        filename, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

    # shuffle here like cf-nade paper with python's own random class
    # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
    
    data_array = data.to_numpy()
    random.seed(seed)
    random.shuffle(data_array)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)
    user_feat = torch.load(f'graph/saved/init_embedding/{filename}/user_feat.pt')
    item_feat = torch.load(f'graph/saved/init_embedding/{filename}/item_feat.pt')
    return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, user_feat, item_feat


def create_trainvaltest_split(dataset, seed=1234, testing=False, datasplit_path=None, datasplit_from_file=False,
                              verbose=True):

    if datasplit_from_file and os.path.isfile(datasplit_path):
        print('Reading dataset splits from file...')
        with open(datasplit_path) as f:
            num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = pkl.load(f)

    else:
        num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = load_data(dataset, seed=seed,
                                                                                            verbose=verbose)

        with open(datasplit_path, 'wb') as f:
            pkl.dump([num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features], f)

    neutral_rating = -1

    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
    labels = labels.reshape([-1])

    # number of test and validation edges
    num_test = int(np.ceil(ratings.shape[0] * 0.1))
    if dataset == 'ml-100k':
        num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))
    else:
        num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))

    num_train = ratings.shape[0] - num_val - num_test

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])

    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    train_idx = idx_nonzero[0:num_train]

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    class_values = np.sort(np.unique(ratings))

    return u_features, v_features, rating_mx_train, class_values

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ---------- Only for test --------------
def sample_batch(num_users, batch_size):
    if isinstance(num_users, int):
        tensor = [list(range(num_users))]
    if len(tensor) == 1:
        tensor = tensor[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i: i + batch_size]
    else:
        for i in range(0, len(tensor), batch_size):
            yield tuple([x[i: i + batch_size] for x in tensor])

import torch.nn.functional as F

def softmax_cross_entropy(input, target):
    """ computes average softmax cross entropy """

    input = input.view(input.size(0),-1).t()
    target = target.view(target.size(0),-1).t()

    omg = torch.sum(target,1).detach()
    len_omg = len(torch.nonzero(omg))
    target = torch.max(target, 1)[1]

    loss = F.cross_entropy(input=input, target=target, reduction='none')
    loss = torch.sum(torch.mul(omg, loss))/len_omg

    return loss

def avg_softmax_cross_entropy(input, target):
    loss = F.cross_entropy(input, target, reduction='none') 
    average_loss = torch.mean(loss)
    return average_loss

def rmse(logits, labels):

    omg = torch.sum(labels, 0).detach()
    len_omg = len(torch.nonzero(omg))

    pred_y = logits
    y = torch.max(labels, 0)[1].float() + 1.

    se = torch.sub(y, pred_y).pow_(2)
    mse= torch.sum(torch.mul(omg, se))/len_omg
    rmse = torch.sqrt(mse + 1e-8)

    return rmse

def rmse(predictions, targets):
    squared_errors = (predictions - targets) ** 2
    rmse = torch.sqrt(torch.mean(squared_errors))
    return rmse

def calculate_loss(u_features, v_features, recon_u, recon_v, mask, score):
    loss = avg_softmax_cross_entropy(recon_u, u_features) + avg_softmax_cross_entropy(recon_v, v_features)
    
    loss += rmse(recon_u[mask], u_features[mask])
    loss -= torch.sum(score) / len(mask)
    return loss
    