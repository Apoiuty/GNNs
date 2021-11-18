from torch_geometric.datasets import TUDataset


def dataset_list(root, *args):
    """
    返回数据集列表
    :param root: 数据集保存目录
    :param args: 数据集名称列表
    :return:
    """
    datasets = []
    for dataset in args:
        datasets.append(TUDataset(root, dataset))
    return datasets
