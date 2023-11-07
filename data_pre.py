import os, torch
import numpy as np
from termcolor import cprint
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from scipy import sparse
from tqdm import tqdm
from utils import *


class DataBuilder:
    def __init__(self, path, device):
        super(DataBuilder, self).__init__()
        dataPath_train = path + '/train.txt'
        dataPath_test = path + '/test.txt'

        self.trainMatrix, self.n_users, self.m_items = self.load_matrix(dataPath_train)
        self.numpy_matrix_train = self.trainMatrix
        self.trainMatrix = torch.tensor(self.trainMatrix, dtype=torch.float32).to(device)

        self.testMatrix, _, _ = self.load_matrix(dataPath_test)
        self.numpy_matrix_test = self.testMatrix
        self.testMatrix = torch.tensor(self.testMatrix, dtype=torch.float32).to(device)

        self.construct()

    def construct(self):
        # for train data
        datalist = self.TransformMatrix2DataList(self.numpy_matrix_train)
        self.trainDataSize = len(datalist)
        trainUser, trainItem = [], []
        for interact in datalist:
            src, trg = interact[0], interact[1]
            trainUser.append(src)
            trainItem.append(trg)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        # for test data
        datalist = self.TransformMatrix2DataList(self.numpy_matrix_test)
        self.testDataSize = len(datalist)
        testUser, testItem = [], []
        for interact in datalist:
            src, trg = interact[0], interact[1]
            testUser.append(src)
            testItem.append(trg)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix(self.numpy_matrix_train)
        # 每个user的交互数量
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        # 每个item的交互数量
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # 每个user的交互记录
        self.allPos = self.getUserPosItems(list(range(self.n_users)))
        # user-list ： item-list
        self.testDict = self.__build_test()

    def load_matrix(self, dataPath):
        user_id_list, dataDict = [], {}
        n_users, n_items = -1, -1
        with open(dataPath, 'r') as f:
            for line in f.readlines():
                if line == '' or line is None:
                    break
                l = line.strip('\n').split(' ')
                user_id = int(l[0])
                items_id = [int(i) for i in l[1:]]
                n_items = max(n_items, max(items_id))
                user_id_list.append(user_id)
                dataDict[l[0]] = items_id
        n_users = max(user_id_list)
        dataMatrix = np.zeros((n_users + 1, n_items + 1))
        for user_id, items_id in dataDict.items():
            dataMatrix[int(user_id), items_id] = 1.0
        return dataMatrix, n_users + 1, n_items + 1

    def TransformMatrix2DataList(self, matrix):
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.numpy()
        sparse_matrix = sparse.coo_matrix(matrix)
        row = sparse_matrix.row
        column = sparse_matrix.col
        rating = sparse_matrix.data
        return list(zip(row, column, rating))


    def updata_dataset(self, fake_data):
        numpy_fake_data = fake_data.clone().detach().cpu().numpy()
        self.numpy_matrix_train = np.concatenate((self.numpy_matrix_train, numpy_fake_data), axis=0)
        self.n_users += numpy_fake_data.shape[0]
        self.trainMatrix = torch.concat((self.trainMatrix, fake_data), dim=0)
        self.construct()

    def update_graph(self, fake_data, device):
        fake_data = fake_data.to(device)
        numpy_fake_data = fake_data.clone().detach().cpu().numpy()
        self.numpy_matrix_train = np.concatenate((self.numpy_matrix_train, numpy_fake_data), axis=0)
        self.n_users += numpy_fake_data.shape[0]
        self.trainMatrix = torch.concat((self.trainMatrix, fake_data), dim=0)
        self.construct()

        self.Graph = self.getSparseGraph()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self, device):
        # build tensor graph
        # linked sparse matrix(tolil will be much faster than dok matrix/coo matrix)

        # adj_matrix = sparse.coo_matrix(self.trainMatrix, dtype=np.float32)
        # rowsum = np.array(adj_matrix.sum(axis=1))
        # d_inv = np.power(rowsum, -0.5).flatten()
        # d_inv[np.isinf(d_inv)] = 0.
        # d_mat = sp.diags(d_inv)
        #
        # norm_adj = d_mat.dot(adj_matrix)
        # norm_adj = norm_adj.dot(d_mat)
        #
        # indices = torch.from_numpy(
        #     np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64)
        # )
        # values = torch.from_numpy(norm_adj.data)
        # shape = torch.Size(norm_adj.shape)
        #
        #
        # return torch.sparse.FloatTensor(indices, values, shape)
        # -------------------------------------------------------------------------------------
        # adj_mat = sparse.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items),
        #                             dtype=np.float32).tolil()
        # R = self.UserItemNet.tolil()
        # adj_mat[:self.n_users, self.n_users:] = R
        # adj_mat[self.n_users:, :self.n_users] = R.T
        # adj_mat = adj_mat.todok()
        #
        # rowsum = np.array(adj_mat.sum(axis=1))
        # d_inv = np.power(rowsum, -0.5).flatten()
        # d_inv[np.isinf(d_inv)] = 0.
        # d_mat = sp.diags(d_inv)
        #
        # norm_adj = d_mat.dot(adj_mat)
        # norm_adj = norm_adj.dot(d_mat)
        # norm_adj = norm_adj.tocsr()
        #
        # self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        # self.Graph = self.Graph.coalesce().to(config.DEVICE)
        # -------------------------------------------------------------------------------------
        assert isinstance(self.trainMatrix, torch.Tensor), 'only support tensor matrix'
        n, m = self.trainMatrix.shape
        row = torch.concat((torch.zeros(n, n).to(device), self.trainMatrix), dim=1)
        col = torch.concat((self.trainMatrix.T, torch.zeros(m, m).to(device)), dim=1)
        graph = torch.concat((row, col), dim=0).to_sparse()

        nor_graph = self.normalize_tensor(graph)
        self.Graph = nor_graph
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def GetPopularItem(self, p=0.01):
        arg_popular = np.argsort(self.items_D)[::-1]
        hot = int(p * self.m_items)
        return arg_popular[:hot]

    def GetColdItem(self, p=0.05):
        arg_popular = np.argsort(self.items_D)
        cold = int(p * self.m_items)
        return arg_popular[-cold:]


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

    # calculate feature
    # Movie features (genres)
    sep = r'|'
    movie_file = data_dir + files[1]
    movie_headers = ['movie id', 'movie title', 'release date', 'video release date',
                        'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                        'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                        'Thriller', 'War', 'Western']
    movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                            names=movie_headers, engine='python', encoding = 'latin-1')

    genre_headers = movie_df.columns.values[6:]
    num_genres = genre_headers.shape[0]

    v_features = np.zeros((num_items, num_genres), dtype=np.float32)
    for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df[genre_headers].values.tolist()):
        # Check if movie_id was listed in ratings file and therefore in mapping dictionary
        if movie_id in v_dict.keys():
            v_features[v_dict[movie_id], :] = g_vec

    # User features

    sep = r'|'
    users_file = data_dir + files[2]
    users_headers = ['user id', 'age', 'gender', 'occupation', 'zip code']
    users_df = pd.read_csv(users_file, sep=sep, header=None,
                            names=users_headers, engine='python', encoding = 'latin-1')

    occupation = set(users_df['occupation'].values.tolist())

    gender_dict = {'M': 0., 'F': 1.}
    occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}

    num_feats = 2 + len(occupation_dict)

    u_features = np.zeros((num_users, num_feats), dtype=np.float32)
    for _, row in users_df.iterrows():
        u_id = row['user id']
        if u_id in u_dict.keys():
            # age
            u_features[u_dict[u_id], 0] = row['age']
            # gender
            u_features[u_dict[u_id], 1] = gender_dict[row['gender']]
            # occupation
            u_features[u_dict[u_id], occupation_dict[row['occupation']]] = 1.

    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)

    

    # user_feat = torch.load(f'graph/saved/init_embedding/{filename}/user_feat.pt')
    # item_feat = torch.load(f'graph/saved/init_embedding/{filename}/item_feat.pt')
    return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features

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

def get_loader(dataName):

	datasplit_path = 'data/' + dataName + '/features.pickle'

	# 得到user / item
	u_features, v_features, adj_train, class_values = create_trainvaltest_split(dataName, datasplit_path=datasplit_path)

	num_users, num_items = adj_train.shape

	print("Normalizing feature vectors...")

	# node id's for node input features
	id_csr_u = sp.identity(num_users, format='csr')
	id_csr_v = sp.identity(num_items, format='csr')

	u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

	u_features = u_features.toarray()
	v_features = v_features.toarray()

	features_dim = u_features.shape[1]

	return len(class_values), features_dim, u_features, v_features, adj_train


def loadFeatures(dataset, graph, model_path):
    recmodel = model(dataset, graph).cuda()
    recmodel.load_state_dict(torch.load(model_path))
    users_emb, items_emb = recmodel.computer()
    users_emb = users_emb + users_features
    items_emb = items_emb + items_features
    torch.save(users_emb, 'user.pt')
    torch.save(items_emb, 'item.pt')
    return users_emb, items_emb


def init_emb_by_feature(trainMatrix, name=None):
    user_feat, item_feat = None, None
    if os.path.exists(f'saved/init_embedding/{name}/user_feat.pt'):
        user_feat = torch.load(f'saved/init_embedding/{name}/user_feat.pt')
    if os.path.exists(f'/Users/edisonchen/Desktop/graph/saved/init_embedding/{name}/item_feat.pt'):
        item_feat = torch.load(f'/Users/edisonchen/Desktop/graph/saved/init_embedding/{name}/item_feat.pt')
    if user_feat is not None and item_feat is not None:
        return user_feat, item_feat
    feat = Feature(trainMatrix)
    user_feat, item_feat = [], []
    for user in tqdm(range(trainMatrix.shape[0])):
        user_feat.append(feat.get_feature(user))
    user_feat = torch.tensor(user_feat, dtype=torch.float)
    torch.save(user_feat, f'/Users/edisonchen/Desktop/graph/saved/init_embedding/{name}/user_feat.pt')
    feat = Feature(trainMatrix.T)
    for item in tqdm(range(trainMatrix.shape[1])):
        item_feat.append(feat.get_feature(item))
    item_feat = torch.tensor(item_feat, dtype=torch.float)
    torch.save(item_feat, f'/Users/edisonchen/Desktop/graph/saved/init_embedding/{name}/item_feat.pt')
    
    cprint('\n----- loading the init embeddings by dataset features, before senting to the model, it should be scaled -----\n', 'yellow')
    return user_feat, item_feat


if __name__ == '__main__':
    get_loader('ml-100k')