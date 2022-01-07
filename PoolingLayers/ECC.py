import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
import torch_geometric.transforms as T
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader
from torch_geometric.nn import voxel_grid, max_pool


class C(torch.nn.Module):
    def __init__(self, indim, outdim, edge_dim):
        super(C, self).__init__()
        self.mlp = torch.nn.Sequential(
            nn.Linear(edge_dim, indim * outdim),
            nn.ReLU(),
            nn.Linear(indim * outdim, indim * outdim)
        )
        self.net = torch.nn.Sequential(
            tgnn.ECConv(indim, outdim, self.mlp),
            nn.BatchNorm1d(outdim),
            nn.ReLU()
        )

    def forward(self, data):
        x = self.net[0](data.x, data.edge_index, data.edge_attr)
        x = self.net[1](x)
        return self.net[2](x)


class ECC(torch.nn.Module):
    def __init__(self, indim, outdim):
        super(ECC, self).__init__()
        self.conv1 = C(indim, 16, 2)
        self.conv2 = C(16, 32, 2)
        self.conv3 = C(32, 64, 2)
        self.conv4 = C(64, 128, 2)
        self.FC = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
        )

    def forward(self, data):
        data.x = self.conv1(data)
        # 这个是为了得到分组
        cluster = voxel_grid(data.pos, data.batch, size=3.4)
        data.edge_attr = None
        data = max_pool(cluster, data, transform=trans)
        data.x = self.conv2(data)
        cluster = voxel_grid(data.pos, data.batch, size=6.8)
        data.edge_attr = None
        data = max_pool(cluster, data, transform=trans)
        data.x = self.conv3(data)
        cluster = voxel_grid(data.pos, data.batch, size=30)
        data.edge_attr = None
        data = max_pool(cluster, data, transform=trans)
        data.x = self.conv4(data)
        return self.FC(data.x)


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


dataset_path = '../data/MinistG'
trans = T.Cartesian()
train_set = MNISTSuperpixels(dataset_path, True, transform=trans, )
test_set = MNISTSuperpixels(dataset_path, False, transform=trans, )
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

device = 'cpu' if not torch.cuda.is_available() else 'cuda'
model = ECC(train_set.num_features, train_set.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, )
criterion = torch.nn.CrossEntropyLoss()
epoch = 300
best_val_acc = 0
best_test_acc = 0
epoch_to_break = 0
for i in range(epoch):
    train_sum_acc = 0
    train_cnt = 0
    train_loss = 0
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        result = model(data)
        loss = criterion(result, data.y)
        train_loss += loss
        loss.backward()
        optimizer.step()
        pred = result.max(dim=1)[1]
        acc = torch.eq(pred, data.y).sum().item()
        train_sum_acc += acc
        train_cnt += len(data.y)

    val_acc = test(model, test_loader, device)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test(model, test_loader, device)
        epoch_to_break = 0
    else:
        epoch_to_break += 1

    if epoch_to_break >= 20:
        break

    print(
        f'Epoch{i}: TrainAcc: {train_sum_acc / train_cnt:.6f} TrainLoss:{train_loss / train_cnt:.6f} ValAcc:{val_acc:.6f}'
        f' BestTestAcc: {best_test_acc:.6f}')
