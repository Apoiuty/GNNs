import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import dropout_adj


class GUNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, ratio=.5):
        super().__init__()
        self.unet = GraphUNet(in_dim, hidden_dim, out_dim, depth=num_layer, pool_ratios=ratio)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = dropout_adj(edge_index, p=0.2, force_undirected=True, training=self.training)
        x = torch.nn.functional.dropout(x, p=0.92, training=self.training)
        x = self.unet(x, edge_index)
        return x


@torch.no_grad()
def test(model, data, mask):
    """
    测试模型，返回精度
    :param data:
    :param model:
    :param mask:
    :return:
    """
    model.eval()
    result = model(data)[mask]
    target = data.y[mask]
    pred = result.max(dim=1)[1]
    acc = torch.eq(pred, target).sum().item() / len(target)
    return acc, loss


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


set_seed(0)
dataset = 'Cora'
path = osp.join('..', 'data', dataset)
dataset = Planetoid(path, dataset)
data = dataset[0]

total_num = len(data.x)
ratio = [2000 / total_num, .5]
Net = GUNet(dataset.num_features, 64, dataset.num_classes, 3, ratio=ratio)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, data = Net.to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
criterion = torch.nn.CrossEntropyLoss()
train_set = data.train_mask
val_set = data.val_mask
test_set = data.test_mask

epoch = 50
best_val_acc = 0
best_test_acc = 0
epoch_to_break = 0
for i in range(epoch):
    model.train()
    optimizer.zero_grad()
    result = model(data)[train_set]
    target = data.y[train_set]
    loss = criterion(result, target)
    loss.backward()
    optimizer.step()
    pred = result.max(dim=1)[1]

    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)

    train_acc, val_acc, best_test_acc = accs
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epoch_to_break = 0
    else:
        epoch_to_break += 1

    if epoch_to_break >= 1000:
        break

    print(
        f'Epoch{i}: TrainAcc: {train_acc:.6f}  ValAcc:{val_acc:.6f} BestTestAcc: {best_test_acc:.6f}')
