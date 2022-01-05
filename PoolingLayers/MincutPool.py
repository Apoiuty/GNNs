from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import DenseSAGEConv, dense_mincut_pool, DenseGraphConv
from torch_geometric.nn import GCNConv
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


class MincutPool(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        num_nodes = ceil(0.5 * average_nodes)
        self.pool1 = Linear(hidden_channels, num_nodes)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = Linear(hidden_channels, num_nodes)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        s = self.pool1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = F.relu(self.conv2(x, adj))
        s = self.pool2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

        x = self.conv3(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x, mc1 + mc2 + o1 + o2


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
        pred = model(data.x, data.edge_index, data.batch)[0].max(dim=1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


# DiffPool在ENZYMES的复现
input_path = '../data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TUDataset(input_path, 'PROTEINS')
average_nodes = 40
train_cnt = int(.8 * len(dataset))
val_cnt = int(.1 * len(dataset))
test_cnt = len(dataset) - train_cnt - val_cnt
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_cnt, val_cnt, test_cnt],
                                                                torch.random.manual_seed(0))
max_node = 0
for data in dataset:
    if data.num_nodes > max_node:
        max_node = data.num_nodes

model = MincutPool(dataset.num_features, dataset.num_classes, hidden_channels=16).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
        x, adj, mask = data.x, data.edge_index, data.batch
        result, mincut_loss = model(x, adj, mask)
        loss = criterion(result, data.y) + mincut_loss
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
