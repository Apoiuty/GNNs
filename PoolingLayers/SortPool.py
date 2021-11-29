import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_sort_pool


class SortPool(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k):
        super().__init__()
        self.k = k
        self.conv = nn.ModuleList()
        for i in range(4):
            if i == 0:
                self.conv.add_module(f'Conv{i}', GCNConv(in_dim, hidden_dim))
            elif i == 3:
                self.conv.add_module(f'Conv{i}', GCNConv(hidden_dim, 1))
            else:
                self.conv.add_module(f'Conv{i}', GCNConv(hidden_dim, hidden_dim))

        self.conv1 = nn.Conv1d(in_channels=hidden_dim * 3 + 1, out_channels=16, kernel_size=1, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.outdim = out_dim

    def forward(self, x, edge_index, batch):
        x_list = []
        for conv in self.conv:
            x = conv(x, edge_index)
            x = torch.tanh(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=-1)
        x = global_sort_pool(x, batch, self.k)
        x = x.view(len(x), self.k, -1).permute([0, 2, 1])
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = x.view(len(x), -1)
        if hasattr(self, 'mlp'):
            return self.mlp(x)
        else:
            mlp_indim = x.shape[-1]
            self.mlp = nn.Sequential(
                nn.Linear(mlp_indim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.outdim)
            )
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
dataset = TUDataset(input_path, 'DD')
hidden_dim = 32
train_cnt = int(.8 * len(dataset))
val_cnt = int(.1 * len(dataset))
test_cnt = len(dataset) - train_cnt - val_cnt
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_cnt, val_cnt, test_cnt],
                                                                torch.random.manual_seed(0))

node_list = []
for data in dataset:
    node_list.append(data.num_nodes)
node_list.sort()
k = node_list[int(len(node_list) * .4)]

model = SortPool(dataset.num_features, hidden_dim, dataset.num_classes, k).to(device)
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
