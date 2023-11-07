import os
import utils
import torch
import graph
import random
import importlib
import yaml
from config import get_args
import torch.optim as optim
from data_pre import get_loader
from tqdm import tqdm
import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler

args = get_args()
print(args)
# config_path = '2022_03_22.yaml'
# with open(config_path, "r", encoding="utf8") as fr:
#     args = yaml.safe_load(fr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
recommender = importlib.import_module(args.recommender)
trainer = recommender.Trainer(dataset=args.dataName, target_item=args.target)
result = trainer.fit()

# Load the datas
num_classes, num_features, u_features, v_features, trainMatrix = get_loader(args.dataName)

num_users, num_items = trainMatrix.shape

u_features = torch.from_numpy(u_features).to(device).float()
v_features = torch.from_numpy(v_features).to(device).float()
trainMatrix = torch.from_numpy(trainMatrix).to(device).float()
feat_dim = u_features.shape[1]

ratings = torch.load(args.train_path).to(device)
attacker = graph.SubGraphGen(u_features, v_features, num_classes, trainMatrix, args)

if torch.cuda.is_available():
    attacker.cuda()
"""Print out the network information."""
num_params = 0
for p in attacker.parameters():
    num_params += p.numel()
print("The number of parameters: {}".format(num_params))

torch.autograd.set_detect_anomaly(True)
def train(recommender, 
          u_features,
          v_features,
          num_users,
          args):

    attacker_pretrain_path = f'./saved/attacker/{args.dataName}_{args.recommender.split(".")[-1]}.pt'
    optimizer = optim.Adam(attacker.parameters(), lr = args.lr, betas=[args.beta1, args.beta2])
    num_steps = args.epochs * (num_users // args.batch_size + 1)
    processbar = tqdm(range(args.epochs))

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=args.lr,
        cycle_limit=1,
        t_in_epochs=False,
    )

    if not os.path.exists(attacker_pretrain_path):
        os.makedirs("./saved/attacker/", exist_ok=True)
        for epoch in processbar:
            losses, idx = [], 0
            processbar.set_description(f'Epoch: {epoch}')
            for mask in utils.sample_batch(num_users, batch_size=args.batch_size):
                random.shuffle(mask)
                recon_u, recon_v, score, edges = attacker(args.target, 
                                                          args.attack_size, 
                                                          args.budget, 
                                                          args.n_ran,
                                                          args.mode)
                graph_loss = utils.calculate_loss(u_features, v_features, recon_u, recon_v, mask, score)
                if epoch >= args.limit:
                    trainer = recommender.Trainer(dataset=args.dataName, target_item=args.target)
                    adv_loss = trainer.fit_adv(edges, args.target, args.ratio)
                    loss = graph_loss + adv_loss
                else:
                    loss = graph_loss
                print(adv_loss, graph_loss)
                # with torch.autograd.detect_anomaly():
                loss.backward()
                optimizer.zero_grad()
                optimizer.step()
                lr_scheduler.step_update(args.epochs * num_steps + idx)
                idx += 1
                losses.append(loss.item())
            processbar.set_postfix_str(f"loss: {np.mean(losses)}")

    else:
        attacker.load_state_dict(torch.load(attacker_pretrain_path))

train(recommender, u_features, v_features, num_users, args)