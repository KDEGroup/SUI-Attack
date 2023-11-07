import random, os, shutil
import numpy as np
import torch, scipy
from easydict import EasyDict as edict
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras.layers import Input, Dense, Lambda
from tensorflow.compat.v1.keras.models import Sequential, Model
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.regularizers import l2 as L2
seed = 3407

import time, math
tf.disable_v2_behavior()
tf.set_random_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import sys
sys.path.append('/Users/edisonchen/Desktop/graph')
from models.recommender.WMF import WMFTrainer

def save_fake(path_save_fake, item_list, shape, results_dir='/Users/edisonchen/Desktop/graph/models/baseline/results/injected'):
        print(f"[SAVE FakeData in attacker] Writing fake users to: {results_dir}")
        if not os.path.exists(results_dir):
            print(f"[SAVE FakeData in attacker] Creating output root directory {results_dir}")
            os.makedirs(results_dir)

        row, col, rating = [], [], []
        for idx, items in enumerate(item_list):
            for item in items:
                row.append(idx)
                col.append(item)
                rating.append(1.0)
        matrix = sparse.csr_matrix((rating, (row, col)), shape=shape)
        path = os.path.join(results_dir, path_save_fake + '.npz')
        sparse.save_npz(path, matrix)
        return path

class DataLoader(object):

    def __init__(self, path_train, path_test, header=None, sep='\t', threshold=4, verbose=False):
        self.path_train = path_train
        self.path_test = path_test
        self.header = header if header is not None else ['user_id', 'item_id', 'rating']
        self.sep = sep
        self.threshold = threshold
        self.verbose = verbose

        # load file as dataFrame
        # self.train_data, self.test_data, self.n_users, self.n_items = self.load_file_as_dataFrame()
        # dataframe to matrix
        # self.train_matrix, self.train_matrix_implicit = self.dataFrame_to_matrix(self.train_data)
        # self.test_matrix, self.test_matrix_implicit = self.dataFrame_to_matrix(self.test_data)

    def load_file_as_dataFrame(self):
        # load data to pandas dataframe
        if self.verbose:
            print("\nload data from %s ..." % self.path_train, flush=True)

        train_data = pd.read_csv(self.path_train, sep=self.sep, names=self.header, engine='python')
        train_data = train_data.loc[:, ['user_id', 'item_id', 'rating']]

        if self.verbose:
            print("load data from %s ..." % self.path_test, flush=True)
        test_data = pd.read_csv(self.path_test, sep=self.sep, names=self.header, engine='python').loc[:,
                    ['user_id', 'item_id', 'rating']]
        test_data = test_data.loc[:, ['user_id', 'item_id', 'rating']]

        # data statics

        n_users = max(max(test_data.user_id.unique()), max(train_data.user_id.unique())) + 1
        n_items = max(max(test_data.item_id.unique()), max(train_data.item_id.unique())) + 1

        if self.verbose:
            print("Number of users : %d , Number of items : %d. " % (n_users, n_items), flush=True)
            print("Train size : %d , Test size : %d. " % (train_data.shape[0], test_data.shape[0]), flush=True)

        return train_data, test_data, n_users, n_items

    def dataFrame_to_matrix(self, data_frame, n_users, n_items):
        row, col, rating, implicit_rating = [], [], [], []
        for line in data_frame.itertuples():
            uid, iid, r = list(line)[1:]
            implicit_r = 1 if r >= self.threshold else 0

            row.append(uid)
            col.append(iid)
            rating.append(r)
            implicit_rating.append(implicit_r)

        matrix = csr_matrix((rating, (row, col)), shape=(n_users, n_items))
        matrix_implicit = csr_matrix((implicit_rating, (row, col)), shape=(n_users, n_items))
        return matrix, matrix_implicit


class Attacker(object):
    def __init__(self, parmas):
        self.data_set = parmas.data_set
        self.target_id = parmas.target_id
        self.attack_num = parmas.attack_num

        self.filler_num = 36
        self.injected_path = parmas.injected_path
        self.model_path = f'/Users/edisonchen/Desktop/graph/models/baseline/results/model_saved/{parmas.attacker}_{self.data_set}_{self.target_id}_{self.attack_num}'
        self.cuda_id = 0

    def prepare_data(self):
        print('----load data -----')
        self.path_train = f'/Users/edisonchen/Desktop/graph/data/{self.data_set}/preprocess/train.data'
        path_test = f'/Users/edisonchen/Desktop/graph/data/{self.data_set}/preprocess/test.data'
        self.dataset = DataLoader(self.path_train, path_test)
        self.train_data_df, _, self.n_users, self.n_items = self.dataset.load_file_as_dataFrame()

    def build_network(self):
        raise NotImplemented

    def train(self):
        raise NotImplemented

    def test(self, victim='SVD', detect=False, fake_array=None):

        fake_array = self.generate_injectedFile(fake_array)

        """attack"""
        all_victim_models = ['WMF']#, 'ItemAE', 'VAE', 'NGCF', 'ItemCF']
        
        victim_models = all_victim_models.split(',')
        res_attack_list = []

        for victim_model in victim_models:
            print('attacking all victim models')
            self.attack(victim_model)
            cur_res_list = self.evaluate(victim_model)

            res_attack_list.append('\t:\t'.join([victim_model, '\t'.join(cur_res_list)]))
        res_attack = ''.join(res_attack_list)
        res = '\t'.join([res_attack])
        return res

    def evaluate(self, victim):
        attacker, recommender = self.__class__.__name__, victim
        # #
        args_dict = {
            'data_set': self.data_set,
            'test_path': './data/%s/%s_test.dat' % (self.data_set, self.data_set),
            #
            'target_ids': self.target_id,
            'recommender': recommender,
            'attacker': attacker,
            #
        }
        #
        path_res_before_attack = f'./results/performance/mid_results/{self.data_set}/{self.data_set}_{recommender}_{self.target_id}_{self.attack_num}.npy'

        if not os.path.exists(path_res_before_attack):
            print("path not exists", path_res_before_attack)
            params = edict({
                'train_path': './data/%s/%s_train.dat' % (self.data_set, self.data_set),
                'model_path': f'./results/model_saved/%s/%s_%s_{self.attack_num}' % (self.data_set, self.data_set, recommender),
                'target_prediction_path_prefix': f'./results/performance/mid_results/{self.data_set}/{self.data_set}_{recommender}_{self.target_id}_{self.attack_num}',
            })
            params.update(args_dict)
            rs = select_recommender(recommender, params)
            rs.execute()

        evaluate_params = edict({
            'data_path_clean': f'./results/performance/mid_results/{self.data_set}/{self.data_set}_{recommender}_{self.target_id}_{self.attack_num}.npy',
            'data_path_attacked': f'./results/performance/mid_results/{self.data_set}/{self.data_set}_{recommender}_{attacker}_{self.target_id}_{self.attack_num}.npy',
            'data_set': self.data_set,
            'recommender': recommender,
            'attacker': attacker,
            'target_id': self.target_id,
            'attack_num': self.attack_num
        })
        evaluate_params.update(args_dict)
        evalute = evaluator.Attack_Effect_Evaluator(evaluate_params)
        result = evalute.execute()
        return result


    def attack(self, victim):
        attacker, recommender = self.__class__.__name__, victim
        print(f'---------------------- {attacker} attacking {recommender} --------------------')
        args_dict = edict({
            #
            'data_set': self.data_set,
            'train_path': f'C:/Users/edison/Desktop/Shilling/ShillingAttack/Leg-UP/results/data_attacked/{self.data_set}/{attacker}_{self.data_set}_{self.target_id}_{self.attack_num}.data',
            'test_path': f'C:/Users/edison/Desktop/Shilling/ShillingAttack/Leg-UP/data/{self.data_set}/{self.data_set}_test.dat',
            #
            'target_ids': self.target_id,
            'recommender': recommender,
            'attacker': attacker,
            #
            'model_path': f'C:/Users/edison/Desktop/Shilling/ShillingAttack/Leg-UP/results/model_saved/{self.data_set}/{self.data_set}_{recommender}_{attacker}_{self.target_id}_{self.attack_num}',
            'target_prediction_path_prefix': f'C:/Users/edison/Desktop/Shilling/ShillingAttack/Leg-UP/results/performance/mid_results/{self.data_set}/{self.data_set}_{recommender}_{attacker}_{self.target_id}_{self.attack_num}',
        })

        target_file = "%s.npy" % (args_dict['target_prediction_path_prefix'])
        if os.path.exists(target_file):
            os.remove(target_file)

        rs = select_recommender(recommender, args_dict)
        rs.execute()



    def execute(self):
        raise NotImplemented

    def save(self, path):
        raise NotImplemented

    def restore(self, path):
        raise NotImplemented

    def generate_fakeMatrix(self):
        raise NotImplemented

    def generate_injectedFile(self, fake_array=None):
        if fake_array is None:
            fake_array = self.generate_fakeMatrix()

        print('fake array shape: ', fake_array.shape)
        if os.path.exists(self.injected_path):
            os.remove(self.injected_path)
        shutil.copyfile(self.path_train, self.injected_path)

        #
        uids = np.where(fake_array > 0)[0] + self.n_users
        iids = np.where(fake_array > 0)[1]
        values = fake_array[fake_array > 0]
        #
        data_to_write = np.concatenate([np.expand_dims(x, 1) for x in [uids, iids, values]], 1)
        F_tuple_encode = lambda x: '\t'.join(map(str, [int(x[0]), int(x[1]), x[2]]))
        data_to_write = '\n'.join([F_tuple_encode(tuple_i) for tuple_i in data_to_write])
        with open(self.injected_path, 'a+')as fout:
            fout.write(data_to_write)

        return fake_array


class AUSH(Attacker):

    def __init__(self, params):
        super(AUSH, self).__init__(params)
        self.selected_ids = [1, 2, 3]
        self.restore_model = 0
        self.epochs = 20
        self.batch_size = 256
        self.learning_rate_G = 0.01
        self.reg_rate_G = 0.0001
        self.ZR_ratio = 0.2
        self.learning_rate_D = 0.001
        self.reg_rate_D = 1e-5
        self.verbose = 1
        self.T = 5

    def prepare_data(self):
        super(AUSH, self).prepare_data()
        train_matrix, _ = self.dataset.dataFrame_to_matrix(self.train_data_df, self.n_users, self.n_items)
        self.train_data_array = train_matrix.toarray()
        self.train_data_mask_array = scipy.sign(self.train_data_array)

        mask_array = (self.train_data_array > 0).astype(np.float)
        mask_array[:, self.selected_ids + [self.target_id]] = 0
        self.template_idxs = np.where(np.sum(mask_array, 1) >= self.filler_num)[0]

    def build_network(self):
        optimizer_G = Adam(learning_rate=self.learning_rate_G)
        optimizer_D = Adam(learning_rate=self.learning_rate_D)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer_D,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()
        real_profiles = Input(shape=(self.n_items,))
        fillers_mask = Input(shape=(self.n_items,))
        selects_mask = Input(shape=(self.n_items,))
        target_patch = Input(shape=(self.n_items,))
        fake_profiles = self.generator([real_profiles, fillers_mask, selects_mask, target_patch])
        self.discriminator.trainable = False
        dis_input = keras.layers.multiply([fake_profiles, keras.layers.add([selects_mask, fillers_mask])])
        output_validity = self.discriminator(dis_input)

        def custom_generator_loss(input_template, output_fake, output_validity, ZR_mask, selects_mask):
            loss_shilling = Lambda(lambda x: keras.backend.mean(
                (x * selects_mask - selects_mask * 5.) ** 2,
                axis=-1, keepdims=True))(output_fake)
            loss_reconstruct = Lambda(lambda x: keras.backend.mean(
                ((x * selects_mask - selects_mask * input_template) * ZR_mask) ** 2,
                axis=-1, keepdims=True))(output_fake)
            loss_adv = Lambda(lambda x: keras.backend.binary_crossentropy(
                tf.ones_like(x), x))(output_validity)
            return keras.layers.add([loss_reconstruct, loss_shilling, loss_adv])

        ZR_mask = Input(shape=(self.n_items,))
        self.generator_train = Model(inputs=[real_profiles, fillers_mask, selects_mask, target_patch, ZR_mask],
                                     outputs=custom_generator_loss(real_profiles, fake_profiles, output_validity,
                                                                   ZR_mask, selects_mask))

        self.generator_train.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer_G)

    def build_generator(self):
        reg_rate = self.reg_rate_G
        model = Sequential(name='generator')
        model.add(Dense(units=128, input_dim=self.n_items,
                        activation='sigmoid', use_bias=True,
                        ))
        model.add(Dense(units=self.n_items,
                        activation='sigmoid', use_bias=True,
                        ))
       

        model.add(Lambda(lambda x: x * 5.0, name='gen_profiles'))

        model.summary()

        real_profiles = Input(shape=(self.n_items,))
        fillers_mask = Input(shape=(self.n_items,))
        input_template = keras.layers.Multiply()([real_profiles, fillers_mask])  # real_profiles * fillers_mask
        gen_output = model(input_template)

        selects_mask = Input(shape=(self.n_items,))
        target_patch = Input(shape=(self.n_items,))
        selected_patch = keras.layers.Multiply()([gen_output, selects_mask])
        output_fake = keras.layers.add([input_template, selected_patch, target_patch])
        return Model([real_profiles, fillers_mask, selects_mask, target_patch], output_fake)

    def build_discriminator(self):
        reg_rate = self.reg_rate_D
        model = Sequential(name='discriminator')

        model.add(Dense(units=150, input_dim=self.n_items,
                        activation='sigmoid', use_bias=True,
                        kernel_initializer='random_uniform',
                        bias_initializer='random_uniform',
                        bias_regularizer=L2(reg_rate),
                        kernel_regularizer=L2(reg_rate)
                        ))
        model.add(Dense(150,
                        activation='sigmoid', use_bias=True,
                        kernel_initializer='random_uniform',
                        bias_initializer='random_uniform',
                        bias_regularizer=L2(reg_rate),
                        kernel_regularizer=L2(reg_rate)))
        model.add(Dense(150,
                        activation='sigmoid', use_bias=True,
                        kernel_initializer='random_uniform',
                        bias_initializer='random_uniform',
                        bias_regularizer=L2(reg_rate),
                        kernel_regularizer=L2(reg_rate)))
        model.add(Dense(units=1,
                        activation='sigmoid', use_bias=True,
                        kernel_initializer='random_uniform',
                        bias_initializer='random_uniform',
                        bias_regularizer=L2(reg_rate),
                        kernel_regularizer=L2(reg_rate)))
        model.summary()

        input_profile = Input(shape=(self.n_items,))
        validity = model(input_profile)
        return Model(input_profile, validity)

    def sample_fillers(self, real_profiles):
        fillers = np.zeros_like(real_profiles)
        filler_pool = set(range(self.n_items)) - set(self.selected_ids) - {self.target_id}

        filler_sampler = lambda x: np.random.choice(size=self.filler_num, replace=False,
                                                    a=list(set(np.argwhere(x > 0).flatten()) & filler_pool))
        sampled_cols = [filler_sampler(x) for x in real_profiles]

        sampled_rows = np.repeat(np.arange(real_profiles.shape[0]), self.filler_num)
        fillers[sampled_rows, np.array(sampled_cols).flatten()] = 1
        return fillers

    def train(self):

        total_batch = math.ceil(len(self.template_idxs) / self.batch_size)
        idxs = np.random.permutation(self.template_idxs)  # shuffled ordering
        #

        d_loss_list, g_loss_list = [], []
        for i in range(total_batch):
            batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            valid_labels = np.ones_like(batch_set_idx)
            fake_labels = np.zeros_like(batch_set_idx)

            real_profiles = self.train_data_array[batch_set_idx, :]
            fillers_mask = self.sample_fillers(real_profiles)
            selects_mask = np.zeros_like(fillers_mask)
            selects_mask[:, self.selected_ids] = 1.
            target_patch = np.zeros_like(fillers_mask)
            target_patch[:, self.selected_ids] = 5.

            fake_profiles = self.generator.predict([real_profiles, fillers_mask, selects_mask, target_patch])

            d_loss_real = self.discriminator.train_on_batch(real_profiles * (fillers_mask + selects_mask), valid_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_profiles * (fillers_mask + selects_mask), fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            ZR_mask = (real_profiles == 0) * selects_mask

            pools = np.argwhere(ZR_mask)
            np.random.shuffle(pools)
            pools = pools[:math.floor(len(pools) * (1 - self.ZR_ratio))]
            ZR_mask[pools[:, 0], pools[:, 1]] = 0
            g_loss = self.generator_train.train_on_batch(
                [real_profiles, fillers_mask, selects_mask, target_patch, ZR_mask],
                valid_labels)

            d_loss_list.append(d_loss)
            g_loss_list.append(g_loss)
        return np.mean(d_loss_list), np.mean(g_loss_list)

    def execute(self):

        self.prepare_data()

        self.build_network()

        if self.restore_model:
            self.restore(self.model_path)
            print("loading done.")

        else:
            for epoch in range(self.epochs):
                d_loss_cur, g_loss_cur = self.train()
                if self.verbose and epoch % self.T == 0:
                    print("epoch:%d\td_loss:%.4f\tg_loss:%.4f" % (epoch, d_loss_cur, g_loss_cur))

            self.save(self.model_path)
            print("training done.")

        self.test(victim='SVD', detect=True)



    def save(self, path):
        return

    def restore(self, path):
        return

    def generate_fakeMatrix(self):
        idx = self.template_idxs[np.random.randint(0, len(self.template_idxs), self.attack_num)]
        real_profiles = self.train_data_array[idx, :]
        fillers_mask = self.sample_fillers(real_profiles)
        selects_mask = np.zeros_like(fillers_mask)
        selects_mask[:, self.selected_ids] = 1.
        target_patch = np.zeros_like(fillers_mask)
        target_patch[:, self.target_id] = 5.

        fake_profiles = self.generator.predict([real_profiles, fillers_mask, selects_mask, target_patch])
        selected_patches = fake_profiles[:, self.selected_ids]
        selected_patches = np.round(selected_patches)
        selected_patches[selected_patches > 5] = 5
        selected_patches[selected_patches < 1] = 1
        fake_profiles[:, self.selected_ids] = selected_patches

        return fake_profiles

    def generate_injectedFile(self, fake_array):
        super(AUSH, self).generate_injectedFile(fake_array)


class HeuristicAttacker(Attacker):
    def __init__(self, params):
        super(HeuristicAttacker, self).__init__(params)

    def prepare_data(self):
        super(HeuristicAttacker, self).prepare_data()

    def generate_fakeMatrix(self):
        raise NotImplemented

    def execute(self):
        self.prepare_data()
        self.test(victim='WMF', detect=True)

    def build_network(self):
        return

    def train(self):
        return

    def save(self, path):
        return

    def restore(self, path):
        return


class RandomAttacker(HeuristicAttacker):
    def __init__(self, params):
        super(RandomAttacker, self).__init__(params)

    def prepare_data(self):
        super(RandomAttacker, self).prepare_data()

        self.global_mean = self.train_data_df.rating.mean()
        self.global_std = self.train_data_df.rating.std()


    def generate_fakeMatrix(self):

        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # padding target score
        fake_profiles[:, self.target_id] = 5
        # padding fillers score
        filler_pool = list(set(range(self.n_items)) - {self.target_id})
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array([filler_sampler([filler_pool, self.filler_num]) for _ in range(self.attack_num)]), (-1))
        sampled_rows = [j for i in range(self.attack_num) for j in [i] * self.filler_num]
        sampled_values = np.random.normal(loc=self.global_mean, scale=self.global_std,
                                          size=(self.attack_num * self.filler_num))
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > 5] = 5
        sampled_values[sampled_values < 1] = 1
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles


class BandwagonAttacker(HeuristicAttacker):
    def __init__(self, params):
        super(BandwagonAttacker, self).__init__(params)
        self.selected_ids = [1, 2, 3]


    def prepare_data(self):
        super(BandwagonAttacker, self).prepare_data()

        self.global_mean = self.train_data_df.rating.mean()
        self.global_std = self.train_data_df.rating.std()

        if len(self.selected_ids) == 0:
            sorted_item_pop_df = self.train_data_df.groupby('item_id').agg('count').sort_values('user_id').index[::-1]
            self.selected_ids = sorted_item_pop_df[:1].to_list()


    def generate_fakeMatrix(self):

        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # padding target score
        fake_profiles[:, self.target_id] = 5
        # padding selected score
        fake_profiles[:, self.selected_ids] = 5
        # padding fillers score
        filler_pool = list(set(range(self.n_items)) - {self.target_id} - set(self.selected_ids))
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array([filler_sampler([filler_pool, self.filler_num]) for _ in range(self.attack_num)]), (-1))
        sampled_rows = [j for i in range(self.attack_num) for j in [i] * self.filler_num]
        sampled_values = np.random.normal(loc=self.global_mean, scale=self.global_std,
                                          size=(self.attack_num * self.filler_num))
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > 5] = 5
        sampled_values[sampled_values < 1] = 1
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles


class SegmentAttacker(HeuristicAttacker):
    def __init__(self, params):
        super(SegmentAttacker, self).__init__(params)
        self.selected_ids = [1, 2, 3]



    def prepare_data(self):
        super(SegmentAttacker, self).prepare_data()
        if len(self.selected_ids) == 0:
            import pandas as pd
            p = './data/%s/%s_selected_items' % (self.data_set, self.data_set)
            data = pd.read_csv(p, sep='\t', names=['target_id', 'selected_ids'], engine='python')
            data.target_id = data.target_id.astype(int)
            selected_ids = data[data.target_id == self.target_id].selected_ids.values[0]
            self.selected_ids = list(map(int, str(selected_ids).split(',')))


    def generate_fakeMatrix(self):

        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # padding target score
        fake_profiles[:, self.target_id] = 5
        # padding selected score
        fake_profiles[:, self.selected_ids] = 5
        # padding fillers score
        filler_pool = list(set(range(self.n_items)) - {self.target_id} - set(self.selected_ids))
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array([filler_sampler([filler_pool, self.filler_num]) for _ in range(self.attack_num)]), (-1))
        sampled_rows = [j for i in range(self.attack_num) for j in [i] * self.filler_num]
        sampled_values = np.ones_like(sampled_rows)
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles

