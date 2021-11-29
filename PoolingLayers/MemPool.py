import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, MemPooling


class MemPool(nn.Module):
    def __init__(self, in_dim, hidden_dim, keys, out_dim):
        """
        :param in_dim: 数据特征维数
        :param hidden_dim:
        :param keys: list,每层Memory Key的个数
        :param out_dim: 数据类别
        :return:
        """
        # GMN实现
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.pools = nn.ModuleList()
        for key in keys:
            self.pools.add_module(f'Key{key}', MemPooling(hidden_dim, hidden_dim, heads=5, num_clusters=key))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        k1_loss = 0
        x_list = []
        for i, pool in enumerate(self.pools):
            if i == 0:
                x, S = pool(x, batch)
            else:
                x, S = pool(x)
            x = F.leaky_relu(x)
            k1_loss += MemPooling.kl_loss(S)
        return self.mlp(torch.squeeze(x)), k1_loss

    def update_keys(self, bool):
        for name, params in self.named_parameters():
            if '.k' in name:
                params.requires_grad = bool


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


# MemPool在DD上复现
input_path = './data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TUDataset(input_path, 'DD')
hidden_dim = 80
train_cnt = int(.8 * len(dataset))
val_cnt = int(.1 * len(dataset))
test_cnt = len(dataset) - train_cnt - val_cnt
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_cnt, val_cnt, test_cnt],
                                                                torch.random.manual_seed(0))

num_experiment = 5
best_acc_sum = 0
for i in range(num_experiment):
    model = MemPool(dataset.num_node_features, hidden_dim, [10, 1], dataset.num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1 ** .5, patience=10, verbose=True, threshold=1e-2, threshold_mode='abs'
    )
    train_set = DataLoader(train_data, shuffle=True, batch_size=20)
    val_set = DataLoader(val_data, batch_size=20)
    test_set = DataLoader(test_data, batch_size=20)
    epoch = 3000
    best_train_acc = 0
    best_val_acc = 0
    best_test_acc = 0
    epoch_to_break = 0

    # 该轮中不更新Keys
    for i in range(epoch):
        train_sum_acc = 0
        train_cnt = 0
        train_loss = 0
        model.train()
        model.update_keys(False)
        for data in train_set:
            data = data.to(device)
            optimizer.zero_grad()
            result, _ = model(data.x, data.edge_index, data.batch)
            loss = criterion(result, data.y)
            train_loss += loss
            loss.backward()
            optimizer.step()
            pred = result.max(dim=1)[1]
            acc = torch.eq(pred, data.y).sum().item()
            train_sum_acc += acc
            train_cnt += len(data.y)

        # 该轮中更新Keys
        loss = 0
        model.update_keys(True)
        for data in train_set:
            data = data.to(device)
            _, k1_loss = model(data.x, data.edge_index, data.batch)
            loss += k1_loss
        optimizer.zero_grad()
        loss = loss / len(train_set)
        loss.backward()
        optimizer.step()

        val_acc = test(model, val_set, device)
        lr_scheduler.step(train_sum_acc / train_cnt)

        # early stopping
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

    best_acc_sum += best_test_acc

print(best_acc_sum / num_experiment)
