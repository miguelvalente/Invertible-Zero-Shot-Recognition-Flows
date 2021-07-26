import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

import wandb
from act_norm import ActNormBijection
from affine_coupling import AffineCoupling
from distributions import (DoubleDistribution, SemanticDistribution)
from permuters import LinearLU, Permuter, Reverse
from toydata import ToyData
from transform import Flow
from utils import make_toy_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run = wandb.init(project='toy_data_zf', entity='mvalente',
                 config=r'config/base_conf.yaml')

config = wandb.config

points_per_sample = 30000
run.config['points_per_sample'] = points_per_sample

toy_data = ToyData(points_per_sample, show_data=False)
generate_all = True


cs = toy_data.cs
cu = toy_data.cu
contexts = torch.vstack((cs, cu))

cs = cs.to(device)
cu = cu.to(device)
contexts = contexts.to(device)

input_dim = 4
context_dim = 2
split_dim = input_dim - context_dim

train_loader = DataLoader(toy_data,
                          batch_size=config['batch_size'],
                          shuffle=True, pin_memory=True)

test_seen = toy_data.x.clone()
test_seen = [t for t in torch.split(test_seen,
                                    points_per_sample)]

visual_distribution = dist.MultivariateNormal(torch.zeros(split_dim).to(device), torch.eye(split_dim).to(device))
semantic_distribution = SemanticDistribution(contexts, torch.ones(context_dim).to(device), (2, 1))

base_dist = DoubleDistribution(visual_distribution, semantic_distribution, input_dim, context_dim)

if config['permuter'] == 'random':
    permuter = lambda dim: Permuter(permutation=torch.randperm(dim, dtype=torch.long).to(device))
elif config['permuter'] == 'reverse':
    permuter = lambda dim: Reverse(dim_size=dim)
elif config['permuter'] == 'manual':
    permuter = lambda dim: Permuter(permutation=torch.tensor([2, 3, 0, 1], dtype=torch.long).to(device))
elif config['permuter'] == 'LinearLU':
    permuter = lambda dim: LinearLU(num_features=dim, eps=1.0e-5)

if config['non_linearity'] == 'relu':
    non_linearity = torch.nn.ReLU()
elif config['non_linearity'] == 'prelu':
    non_linearity = nn.PReLU(init=0.01)
elif config['non_linearity'] == 'leakyrelu':
    non_linearity = nn.LeakyReLU()

transforms = []
for index in range(config['block_size']):
    if config['act_norm']:
        transforms.append(ActNormBijection(input_dim, data_dep_init=True))
    transforms.append(permuter(input_dim))
    transforms.append(AffineCoupling(input_dim, hidden_dims=[2], non_linearity=non_linearity, net=config['net']))

flow = Flow(transforms, base_dist)
flow.train()
flow = flow.to(device)

print(f'Number of trainable parameters: {sum([x.numel() for x in flow.parameters()])}')
run.watch(flow)
optimizer = optim.Adam(flow.parameters(), lr=config['lr'])

epochs = tqdm.trange(1, config['epochs'])

number_samples = 400
for epoch in epochs:
    losses = []
    losses_flow = []
    losses_centr = []
    losses_mmd = []
    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        loss_flow = - flow.log_prob(data, targets).mean() * config['wt_f_l']
        centralizing_loss = flow.centralizing_loss(data, targets, cs.to(device)) * config['wt_c_l']
        mmd_loss = flow.mmd_loss(data, cu.to(device)) * config['wt_mmd_l']
        loss = loss_flow + centralizing_loss + mmd_loss
        loss.backward()
        optimizer.step()

        losses_flow.append(loss_flow.item())
        losses_centr.append(centralizing_loss.item())
        losses_mmd.append(mmd_loss.item())
        losses.append(loss.item())

    if True:
        with torch.no_grad():
            test_data = []
            if generate_all:
                for c_id, c in enumerate(contexts):
                    test_data.append(flow.generation(
                                     torch.hstack((c.repeat(number_samples).reshape(-1, 2),
                                                  flow.base_dist.visual_distribution.sample([number_samples])))))

                test_data = [data.to("cpu").detach().numpy() for data in test_data]
                make_toy_graph(test_data, epoch, "all generated", save=True)

            else:
                test_data = [data[:number_samples] for data in test_seen]
                test_data.append(flow.generation(
                                 torch.hstack((cu.repeat(number_samples).reshape(-1, 2),
                                               flow.base_dist.visual_distribution.sample([number_samples])))))

                test_data = [data.to("cpu").detach().numpy() for data in test_data]

                make_toy_graph(test_data, epoch, "unseen generated", save=True)

    run.log({"loss": sum(losses) / len(losses),
             "loss_flow": sum(losses_flow) / len(losses_flow),  # }, step=epoch)
             "loss_central": sum(losses_centr) / len(losses_centr),  # }, step=epoch)
             "loss_mmd": sum(losses_mmd) / len(losses_mmd)}, step=epoch)

    if loss.isnan():
        print('Nan in loss!')
        Exception('Nan in loss!')
