import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv,GMMConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class DeepVGAE(VGAE):
    def __init__(self, args):
        super(DeepVGAE, self).__init__(encoder=GCNEncoder(args.enc_in_channels,
                                                          args.enc_hidden_channels,
                                                          args.enc_out_channels),
                                       decoder=InnerProductDecoder())
        self.gmm = GMMConv(args.enc_out_channels, args.components_num, dim=2,kernel_size=3)
        self.alpha = args.alpha
        self.alpha = 1.79
        self.num_components = 3
        self.weights = nn.Parameter(torch.Tensor(self.num_components))
        self.reset_parameters()
        self.reset_parameters1()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weights)

    def reset_parameters1(self):
        super().reset_parameters()
        self.gmm.reset_parameters()

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        for i in range(self.num_components):
            weighted_z = self.encode(x, edge_index) * self.weights[i]
        z=weighted_z
        z = self.gmm(z, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def loss(self, x, pos_edge_index, all_edge_index):
        for i in range(self.num_components):

            z = self.encode(x, pos_edge_index)
            pos_loss = -torch.log(self.decoder(z, pos_edge_index) + 1e-5).mean()
            #print(self.decoder(z, pos_edge_index))
            all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
            all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

            neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), pos_edge_index.size(1))
            neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index) + 1e-5).mean()
            kl_loss =1 / x.size(0) * self.kl_loss() * self.weights[i]
        return pos_loss + neg_loss + 0.1*kl_loss
    
 
    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        print('Z')
        print(z.size())
        auc_score, ap_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return auc_score, ap_score
