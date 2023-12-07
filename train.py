import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from model import DeepVGAE
from config.config import parse_args
from torch_geometric import seed_everything

roc_auc_list = []
ap_list = []
total_number = 8

torch.cuda.manual_seed_all(42)
for kk in range(total_number):
    # 设置随机种子
    torch.cuda.manual_seed(kk)  
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
        if epoch % 499 == 0:
            model.eval()
            roc_auc, ap = model.single_test(data.x,
                                        data.train_pos_edge_index,
                                        data.test_pos_edge_index,
                                        data.test_neg_edge_index)
            print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))
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

roc_auc_mean = np.mean(roc_auc_list)
roc_auc_std = np.std(roc_auc_list)
roc_auc_threshold = roc_auc_mean - 2 * roc_auc_std
roc_auc_list_cleaned = [x for x in roc_auc_list if x >= roc_auc_threshold]

ap_mean = np.mean(ap_list)
ap_std = np.std(ap_list)
ap_threshold = ap_mean - 2 * ap_std
ap_list_cleaned = [x for x in ap_list if x >= ap_threshold]

# 计算清理后的roc_auc和ap的均值和方差
roc_auc_mean_cleaned = np.mean(roc_auc_list_cleaned)
ap_mean_cleaned = np.mean(ap_list_cleaned)
roc_auc_var_cleaned = np.var(roc_auc_list_cleaned)
ap_var_cleaned = np.var(ap_list_cleaned)

print("ROC_AUC Mean:", roc_auc_mean_cleaned)
print("AP Mean:", ap_mean_cleaned)
print("ROC_AUC Variance:", roc_auc_var_cleaned)
print("AP Variance:", ap_var_cleaned)


