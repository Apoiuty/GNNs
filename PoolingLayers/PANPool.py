import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import PANConv, PANPooling, global_mean_pool


class PANPool(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, num_layer=3, ratio=.5, L=4):
        super().__init__()
        self.conv = nn.ModuleList()
        self.pool = nn.ModuleList()
        for i in range(num_layer):
            if i == 0:
                self.conv.append(PANConv(in_dim, hidden_dim[i], 4))
            else:
                self.conv.append(PANConv(hidden_dim[i - 1], hidden_dim[i], 4))
            self.pool.append(PANPooling(hidden_dim[i], ratio=ratio))

        mlp_indim = hidden_dim[-1]
        self.FC = nn.Sequential(
            nn.Linear(mlp_indim, mlp_indim // 2),
            nn.ReLU(),
            nn.Linear(mlp_indim // 2, mlp_indim // 4),
            nn.ReLU(),
            nn.Linear(mlp_indim // 4, num_class)
        )

    def forward(self, x, edge_index, batch):
        for conv, pool in zip(self.conv, self.pool):
            x, M = conv(x, edge_index)
            x = F.relu(x)
            x, edge_index, _, batch, *_ = pool(x, M, batch)

        x = global_mean_pool(x, batch)
        return self.FC(x)


@torch.no_grad()
def test(model, loader, device):
    """
    测试模型，返回精度
    :param model:
    :param loader:
    :param device:
    :return:
    """
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).max(dim=1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def set_seed(n):
    torch.manual_seed(n)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # 设置所有设备的随机数发生器

    import random
    random.seed(n)
    # 设置Python的随机数发生器

    import numpy as np
    np.random.seed(n)


# set_seed(42)
dataset = 'PROTEINS'
path = osp.join('..', 'data', dataset)
dataset = TUDataset(path, dataset)

ratio = .5
Net = PANPool(dataset.num_features, [64,64,64], dataset.num_classes, 3, ratio=ratio)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
criterion = torch.nn.CrossEntropyLoss()
train_cnt = int(.8 * len(dataset))
val_cnt = int(.1 * len(dataset))
test_cnt = len(dataset) - train_cnt - val_cnt
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_cnt, val_cnt, test_cnt],
                                                                torch.random.manual_seed(0))

train_set = DataLoader(train_data, shuffle=True, batch_size=32)
val_set = DataLoader(val_data)
test_set = DataLoader(test_data)
epoch = 3000
best_val_acc = 0
best_test_acc = 0
epoch_to_break = 0
for i in range(epoch):
    train_sum_acc = 0
    train_cnt = 0
    train_loss = 0
    model.train()
    for data in train_set:
        data = data.to(device)
        optimizer.zero_grad()
        result = model(data.x, data.edge_index, data.batch)
        loss = criterion(result, data.y)
        train_loss += loss
        loss.backward()
        optimizer.step()
        pred = result.max(dim=1)[1]
        acc = torch.eq(pred, data.y).sum().item()
        train_sum_acc += acc
        train_cnt += len(data.y)

    val_acc = test(model, val_set, device)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test(model, test_set, device)
        epoch_to_break = 0
    else:
        epoch_to_break += 1

    if epoch_to_break >= 200:
        break

    print(
        f'Epoch{i}: TrainAcc: {train_sum_acc / train_cnt:.6f} TrainLoss:{train_loss / train_cnt:.6f} ValAcc:{val_acc:.6f}'
        f' BestTestAcc: {best_test_acc:.6f}')
