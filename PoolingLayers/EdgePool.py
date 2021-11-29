import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, JumpingKnowledge, global_mean_pool, global_max_pool, EdgePooling


class GCN_EdgePool(torch.nn.Module):
    def __init__(self, in_dim, out_dim, ratio):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.pool = EdgePooling(hidden_dim, dropout=.2)

    def forward(self, input):
        x, edge_index, batch, readout = input
        x = F.relu(self.conv(x, edge_index))
        x, edge_index, batch, *_ = self.pool(x, edge_index, batch=batch)
        readout.append(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1))
        return x, edge_index, batch, readout


class EdgePool(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, num_layer=3, ratio=.5):
        super().__init__()
        self.conv_pool_layers = torch.nn.Sequential()
        for i in range(num_layer):
            if i == 0:
                self.conv_pool_layers.add_module(f'conv_pool{i}', GCN_EdgePool(in_dim, hidden_dim, ratio))
            else:
                self.conv_pool_layers.add_module(f'conv_pool{i}', GCN_EdgePool(hidden_dim, hidden_dim, ratio))

        self.readout = JumpingKnowledge(mode='cat')
        self.mlp = nn.Sequential(
            nn.Linear(num_layer * hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_class)
        )

    def forward(self, x, edge_index, batch):
        *_, readout = self.conv_pool_layers((x, edge_index, batch, []))
        x = self.readout(readout)
        return self.mlp(x)


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


# EdgePool在Proteins的复现
input_path = '../data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TUDataset(input_path, 'PROTEINS')
hidden_dim = 64
train_cnt = int(.8 * len(dataset))
val_cnt = int(.1 * len(dataset))
test_cnt = len(dataset) - train_cnt - val_cnt
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_cnt, val_cnt, test_cnt],
                                                                torch.random.manual_seed(0))

model = EdgePool(dataset.num_features, hidden_dim, dataset.num_classes, 3).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_set = DataLoader(train_data, shuffle=True, batch_size=128)
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

    if epoch_to_break >= 100:
        break

    print(
        f'Epoch{i}: TrainAcc: {train_sum_acc / train_cnt:.6f} TrainLoss:{train_loss / train_cnt:.6f} ValAcc:{val_acc:.6f}'
        f' BestTestAcc: {best_test_acc:.6f}')
