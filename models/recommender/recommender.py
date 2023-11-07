import os, time, torch
from collections import OrderedDict
import numpy as np
from scipy import sparse

import utils
from .RsData import LoadData, Dataset
from models.recommender.metrics import *

from torch.utils.data import DataLoader



class Recommender:
    def __init__(self, dataset, target_item, device=0, path_fake_data=None,
                 path_fake_matrix=None, path_fake_array=None,
                 dir_model='./saved/recommender/', topk='1,10,20,50,100'):

        self.path_train = './data/' + dataset + '/preprocess/train.data'
        self.path_test = './data/' + dataset + '/preprocess/test.data'

        self.path_fake_data = path_fake_data
        self.path_fake_matrix = path_fake_matrix
        self.path_fake_array = path_fake_array

        self.dir_model = dir_model
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)

        self.dataset = dataset
        self.target_item = target_item
        self.topk = list(map(int, topk.split(',')))
        self.device = torch.device("cuda:%d" % device if torch.cuda.is_available() else "cpu")

        self.net = None
        self.optimizer = None
        self.path_save = None
        self.golden_metric = 'Recall@50_'
        self.metrics = [PrecisionRecall(k=self.topk), NormalizedDCG(k=self.topk)]

    def prepare_data(self):
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        sep = '\t'

        if self.path_fake_data:
            self.path_train = self.path_fake_data

        self.dataset_class = LoadData(self.path_train, self.path_test, test_bool=True, header=header, sep=sep, type='recommender')
        self.train_df, self.test_df, self.n_users, self.n_items = self.dataset_class.load_file_as_dataFrame()
        _, self.train_matrix = self.dataset_class.dataFrame_to_matrix(self.train_df, self.n_users, self.n_items)
        _, self.test_matrix = self.dataset_class.dataFrame_to_matrix(self.test_df, self.n_users, self.n_items)

        trainset = Dataset(self.train_matrix)
        testset = Dataset(self.test_matrix)
        self.train_loader = DataLoader(dataset=trainset, batch_size=32)
        self.test_loader = DataLoader(dataset=testset, batch_size=32)

        if self.path_fake_matrix or self.path_fake_array:
            if self.path_fake_matrix:
                fake_matrix = sparse.load_npz(self.path_fake_matrix)
                fake_matrix[fake_matrix > 0.0] = 1.0
            else:
                fake_matrix = utils.load_fake_array(self.path_fake_array)
            self.train_matrix = sparse.vstack((self.train_matrix, fake_matrix), format="csr")
        
            self.n_fakes = fake_matrix.shape[0]
            self.n_users += self.n_fakes

    def build_network(self):
        raise NotImplemented

    def train(self):
        raise NotImplemented

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        raise NotImplemented

    def restore(self, path):
        start_epoch, model_state, optimizer_state = utils.load_checkpoint(path)
        self.net.load_state_dict(model_state)
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
        return start_epoch

    def evaluate(self, verbose=True):
        n_rows = self.train_matrix.shape[0]
        n_evaluate_users = self.test_matrix.shape[0]

        total_metrics_len = sum(len(x) for x in self.metrics)
        total_val_metrics = np.zeros([n_rows, total_metrics_len], dtype=np.float32)

        recommendations = self.recommend(self.train_matrix, top_k=100)
        valid_rows = []
        for i in range(n_rows):
            if i >= n_evaluate_users:
                continue
            targets = self.test_matrix[i].indices
            if targets.size <= 0:
                continue

            recs = recommendations[i].tolist()

            metric_results = list()
            for metric in self.metrics:
                result = metric(targets, recs)
                metric_results.append(result)
            total_val_metrics[i, :] = np.concatenate(metric_results)
            valid_rows.append(i)

        total_val_metrics = total_val_metrics[valid_rows]
        avg_val_metrics = (total_val_metrics.mean(axis=0)).tolist()

        ind, result = 0, OrderedDict()
        for metric in self.metrics:
            values = avg_val_metrics[ind: ind + len(metric)]
            if len(values) <= 1:
                result[str(metric)] = round(values[0], 3)
            else:
                for name, value in zip(str(metric).split(','), values):
                    result[name] = round(value, 3)
            ind += len(metric)
        if verbose:
            result_pre, result_recall, result_ndcg = [], [], []
            for k, v in result.items():
                if 'Precision' in k:
                    result_pre.append(round(v, 2))
                elif 'Recall' in k:
                    result_recall.append(round(v, 2))
                elif 'NDCG' in k:
                    result_ndcg.append(round(v, 2))
            top_k_str = ', '.join(list(map(str, self.topk)))
            pre_str = ', '.join(list(map(str, result_pre)))
            recall_str = ', '.join(list(map(str, result_recall)))
            ndcg_str = ', '.join(list(map(str, result_ndcg)))
            print('[Evaluation recommender] topk=[{}]\nprecision=[{}], recall=[{}], ndcg=[{}]'.
                  format(top_k_str, pre_str, recall_str, ndcg_str))
        return result

    def validate(self):
        t1 = time.time()

        n_rows = self.train_matrix.shape[0]
        n_evaluate_users = self.test_matrix.shape[0]

        # Init evaluation results.
        target_item_position = np.zeros([n_rows], dtype=np.int64)
        recommendations = self.recommend(self.train_matrix, top_k=100)

        valid_rows = list()
        target_item_num = 0
        for i in range(n_rows):
            # Ignore augmented users, evaluate only on real users.
            if i >= n_evaluate_users:
                continue
            targets = self.test_matrix[i].indices
            if targets.size <= 0:
                continue
            if self.target_item in self.train_matrix[i].indices:
                continue

            recs = recommendations[i].tolist()

            if self.target_item in recs:
                target_item_position[i] = recs.index(self.target_item)
                target_item_num += 1
            else:
                target_item_position[i] = -1

            valid_rows.append(i)

        target_item_position = target_item_position[valid_rows]
        target_item_position = target_item_position[target_item_position >= 0]

        # Summary evaluation results into a dict.
        result = OrderedDict()
        result['HitUserNum'] = target_item_num
        result['TargetAvgRank'] = round(target_item_position.mean(), 1)

        for cutoff in self.topk:
            result['TargetHR@%d_' % cutoff] = round((target_item_position < cutoff).sum() / len(valid_rows), 3)

            _target_item_position = target_item_position[target_item_position < cutoff]
            result['TargetNDCG@%d_' % cutoff] = round((np.log(2) / np.log(np.array(_target_item_position) + 2)).sum() / len(valid_rows), 3)

        hr_list, ndcg_list = [], []
        for k, v in result.items():
            if 'TargetHR' in k:
                hr_list.append(v)
            if 'TargetNDCG' in k:
                ndcg_list.append(v)
        top_k_str = ', '.join(list(map(str, self.topk)))
        hr_str = ', '.join(list(map(str, hr_list)))
        ndcg_str = ', '.join(list(map(str, ndcg_list)))
        print('[Evaluation recommender after attack][{:.1f} s] topk=[{}]\nHitUserNum=[{}], TargetAvgRank=[{}], TargetHR=[{}], TargetNDCG=[{}]'.
              format(time.time() - t1, top_k_str, result['HitUserNum'], result['TargetAvgRank'], hr_str, ndcg_str))
        return result

    def fit(self):
        self.prepare_data()
        self.build_network()
        self.train()
        # if os.path.exists(self.path_save):
        #     self.restore(self.path_save)

        results_rs = self.evaluate()

        results_atk = self.validate()
        return results_rs, results_atk


        

