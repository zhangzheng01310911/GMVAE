import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from model import DeepVGAE
from config.config import parse_args
import numpy as np

total_number = 10
roc_auc_list = []
ap_list = []

for kk in range(total_number):
    # 设置随机种子
    torch.manual_seed(kk)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    model = DeepVGAE(args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    os.makedirs("datasets", exist_ok=True)
    dataset = Planetoid("datasets", args.dataset, transform=T.NormalizeFeatures())

    # 使用相同的随机种子将数据分割为训练集和测试集
    data = dataset[0].to(device)
    all_edge_index = data.edge_index
    data = train_test_split_edges(data, 0.05, 0.1)

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(data.x, data.train_pos_edge_index, all_edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            model.eval()
            roc_auc, ap = model.single_test(data.x,
                                            data.train_pos_edge_index,
                                            data.test_pos_edge_index,
                                            data.test_neg_edge_index)
            print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))

            # 将roc_auc和ap值添加到列表中
            roc_auc_list.append(roc_auc)
            ap_list.append(ap)

# 计算roc_auc和ap的均值和方差
roc_auc_mean = np.mean(roc_auc_list)
ap_mean = np.mean(ap_list)
roc_auc_var = np.var(roc_auc_list)
ap_var = np.var(ap_list)

print("ROC_AUC Mean:", roc_auc_mean)
print("AP Mean:", ap_mean)
print("ROC_AUC Variance:", roc_auc_var)
print("AP Variance:", ap_var)
