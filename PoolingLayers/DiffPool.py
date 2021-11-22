from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import dense_diff_pool, DenseSAGEConv
from torch_geometric.utils import to_dense_batch, to_dense_adj


class GNN(torch.nn.Module):
    def __init__(self, num_layer, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(num_layer):
            if i == 0:
                self.layers.append(DenseSAGEConv(in_dim, hidden_dim))
            elif i == num_layer - 1:
                self.layers.append(DenseSAGEConv(hidden_dim, out_dim))
            else:
                self.layers.append(DenseSAGEConv(hidden_dim, hidden_dim))

            if i != num_layer - 1:
                self.layers.append(torch.nn.BatchNorm1d(hidden_dim))
            else:
                self.layers.append(torch.nn.BatchNorm1d(out_dim))

    def forward(self, x, a, mask=None):
        for layer in self.layers:
            if isinstance(layer, DenseSAGEConv):
                # 卷积
                x = layer(x, a, mask)
            else:
                # 正则化
                batch_size, num_nodes, num_channels = x.size()
                x = x.view(-1, num_channels)
                x = layer(x)
                x = x.view(batch_size, num_nodes, num_channels)
                x = F.relu(x)
        return x


class DiffPool(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_cluster):
        super().__init__()
        self.embedding = GNN(3, input_dim, hidden_dim, out_dim)
        self.assignment = GNN(3, input_dim, hidden_dim, num_cluster)

    def forward(self, x, a, mask=None):
        z = self.embedding(x, a, mask)
        s = self.assignment(x, a, mask)
        s = F.softmax(s, dim=1)
        return dense_diff_pool(z, a, s, mask)


class Net(torch.nn.Module):
    def __init__(self, dataset, max_node, hidden_dim):
        """
        模型
        :param dataset:
        :param max_node:
        :param hidden_dim:
        """
        super().__init__()
        self.max_node = max_node
        pool1_cluster = ceil(.25 * self.max_node)
        pool2_cluster = ceil(.25 * pool1_cluster)

        self.p1 = DiffPool(dataset.num_features, hidden_dim, hidden_dim, pool1_cluster)
        self.p2 = DiffPool(hidden_dim, hidden_dim, hidden_dim, pool2_cluster)
        self.final_embed = GNN(3, hidden_dim, hidden_dim, hidden_dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim // 2),
                                       torch.nn.BatchNorm1d(hidden_dim // 2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim // 2, dataset.num_classes))

    def forward(self, x, a, mask=None):
        x, a, l1, e1 = self.p1(x, a, mask)
        x, a, l2, e2 = self.p2(x, a)
        x = self.final_embed(x, a)
        x = x.mean(dim=1)
        return self.mlp(x), l1 + l2 + e1 + e2


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
        x, mask = to_dense_batch(data.x, data.batch)
        adj = to_dense_adj(data.edge_index, data.batch)
        pred = model(x, adj, mask)[0].max(dim=1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


# DiffPool在ENZYMES的复现
input_path = '../data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TUDataset(input_path, 'ENZYMES')
train_cnt = int(.8 * len(dataset))
val_cnt = int(.1 * len(dataset))
test_cnt = len(dataset) - train_cnt - val_cnt
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_cnt, val_cnt, test_cnt],
                                                                torch.random.manual_seed(0))
max_node = 0
for data in dataset:
    if data.num_nodes > max_node:
        max_node = data.num_nodes

model = Net(dataset, max_node, 64).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
train_set = DataLoader(train_data, shuffle=True, batch_size=32)
val_set = DataLoader(val_data)
test_set = DataLoader(test_data)
epoch = 300
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
        x, mask = to_dense_batch(data.x, data.batch)
        adj = to_dense_adj(data.edge_index, data.batch)
        result, lpe_loss = model(x, adj, mask)
        loss = criterion(result, data.y) + lpe_loss
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

    if epoch_to_break >= 100:
        break

    print(
        f'Epoch{i}: TrainAcc: {train_sum_acc / train_cnt:.6f} TrainLoss:{train_loss / train_cnt:.6f} ValAcc:{val_acc:.6f}'
        f' BestTestAcc: {best_test_acc:.6f}')
