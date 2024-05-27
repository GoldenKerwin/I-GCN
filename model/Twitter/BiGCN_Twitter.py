import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
from BP import Interaction_GraphConvolution as i_GCNConv
import copy
from torch_geometric.utils import to_undirected
import torch.nn as nn
from torch_scatter import scatter_add


def edges_to_adjacency_matrix(x,edge_index):
    n = x.shape[0]  # 获取节点数量
    adjacency_matrix = np.zeros((n, n))  # 创建零矩阵
    for edge in edge_index:
        i,j = edge[0].item(),edge[1].item()  # 对于每一条边，更新邻接矩阵
        adjacency_matrix[i][j] = 1 
        adjacency_matrix[j][i] = 1 # 无向图的邻接矩阵是对称的
    adjacency_matrix = th.from_numpy(adjacency_matrix).float()  # 将邻接矩阵转换为tensor
    return adjacency_matrix


def normalize_adjacency(x, edge_index):
    """
    对无向图的边进行归一化，不考虑自环，每条边的权重都为1。
    
    参数:
    x (Tensor): 节点特征矩阵。
    edge_index (Tensor): 无向图的边索引，形状为 [2, num_edges]。
    
    返回:
    Tensor: 归一化的邻接矩阵。
    """
    # 确定图中节点的数量
    num_nodes = x.shape[0]
    
    # 移除自环
    self_loops_mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, self_loops_mask]

    # 创建边权重张量
    edge_weight = th.ones((edge_index.size(1), ), dtype=th.float32)

    # 计算每个节点的度
    row, col = edge_index
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

    # 计算度的逆平方根
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # 应用对称归一化
    edge_weight_normalized = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # 创建稀疏的归一化邻接矩阵
    normalized_adjacency_sparse = th.sparse.FloatTensor(edge_index, edge_weight_normalized, th.Size([num_nodes, num_nodes]))
    
    # 转换为密集矩阵
    normalized_adjacency_dense = normalized_adjacency_sparse.to_dense()

    return normalized_adjacency_dense


def caculation(adjacency_matrix):
    num_nodes = adjacency_matrix.size(0)
    adjacency_matrix = adjacency_matrix.float()

    sibling_matrix = th.mm(adjacency_matrix, adjacency_matrix) + adjacency_matrix

    sibling_matrix[range(num_nodes), range(num_nodes)] = 0

    # mask_hadamard = sibling_matrix.unsqueeze(2).expand(-1, num_nodes, -1)

    mask_hadamard = sibling_matrix.unsqueeze(1).expand(-1, 1, -1).float()
    mask_father = adjacency_matrix.unsqueeze(1).expand(-1, 1, -1).float()

    neighbor_count = adjacency_matrix.sum(dim=1, keepdim=True)
    neighbor_count = th.max(neighbor_count, th.ones_like(neighbor_count))

    return mask_father, neighbor_count, mask_hadamard


class DynamicLinear(nn.Module):
    def __init__(self, output_features):
        super(DynamicLinear, self).__init__()
        self.output_features = output_features
        self.linear = None

    def forward(self, x):
        if self.linear is None:
            input_features = x.size(-1)
            self.linear = nn.Linear(input_features, self.output_features)
            self.linear.to(x.device)  
        return self.linear(x)

class TDrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        self.linear = DynamicLinear(in_feats)
        self.conv1 = i_GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+2*in_feats, out_feats)
        # self.conv2 = GCNConv(hid_feats+in_feats, out_feats)


    def forward(self, data, x, edge_index, adj, mask_father, neighbor_count, mask_hadamard):
        x = x.to(device)
        edge_index = edge_index.to(device)
        adj = adj.to(device)
        mask_father = mask_father.to(device)
        neighbor_count = neighbor_count.to(device)
        mask_hadamard = mask_hadamard.to(device)
        
        x = self.linear(x)
        x1 = copy.copy(x.float())
        
        x = self.conv1(x, adj, mask_father, neighbor_count, mask_hadamard)
        x = th.cat((x,x1),1)
        
        x2=copy.copy(x)
        
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch.to(device), num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch.to(device), num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)

        return x


class Net(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.fc=th.nn.Linear((out_feats+hid_feats+in_feats),4)
        # self.fc=th.nn.Linear((out_feats),4)

    def forward(self, data, x, edge_index, adj, mask_father, neighbor_count, mask_hadamard):
        x = self.TDrumorGCN(data, x, edge_index, adj, mask_father, neighbor_count, mask_hadamard)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def train_GCN(treeDic, x_test, x_train,TDdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter):
    model = Net(5000,64,64).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            edge_index = to_undirected(Batch_data.edge_index.to(device))
            x = Batch_data.x.to(device)
            adj = normalize_adjacency(x, edge_index)
            mask_father, neighbor_count, mask_hadamard = caculation(adj)
            out_labels= model(Batch_data, x, edge_index, adj, mask_father, neighbor_count, mask_hadamard)
            finalloss=F.nll_loss(out_labels,Batch_data.y)
            loss=finalloss
            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name} grad: {param.grad.norm()}")
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            val_out = model(Batch_data, x, edge_index, adj, mask_father, neighbor_count, mask_hadamard)
            val_loss  = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', dataname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4

lr=0.0005
weight_decay=1e-4
patience=10
n_epochs=200
#batchsize=128
batchsize=32
TDdroprate=0.2
datasetname=sys.argv[1] #"Twitter15"、"Twitter16"
iterations=int(sys.argv[2])
model="GCN"



device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
test_accs = []
NR_F1 = []
FR_F1 = []
TR_F1 = []
UR_F1 = []
for iter in range(iterations):
    fold0_x_test, fold0_x_train, \
    fold1_x_test,  fold1_x_train,  \
    fold2_x_test, fold2_x_train, \
    fold3_x_test, fold3_x_train, \
    fold4_x_test,fold4_x_train = load5foldData(datasetname)
    treeDic=loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(treeDic,
                                                                                               fold0_x_test,
                                                                                               fold0_x_train,
                                                                                               TDdroprate,
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(treeDic,
                                                                                               fold1_x_test,
                                                                                               fold1_x_train,
                                                                                               TDdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(treeDic,
                                                                                               fold2_x_test,
                                                                                               fold2_x_train,
                                                                                               TDdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(treeDic,
                                                                                               fold3_x_test,
                                                                                               fold3_x_train,
                                                                                               TDdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(treeDic,
                                                                                               fold4_x_test,
                                                                                               fold4_x_train,
                                                                                               TDdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
    NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
    FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
    UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
    sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))


