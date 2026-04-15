import torch.nn as nn
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
import argparse
import time
import torch
from tqdm import tqdm
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, edge_index, x):
        x = self.gc1(x, edge_index)
        return x


def accuracy(output, labels):
    output = output.squeeze()
    preds = (output > 0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def get_model(nfeat, args):
    model = GCN(nfeat, args.num_hidden, args.dropout)
    return model


class FairGNN(nn.Module):

    def __init__(self, nfeat, sim_coeff=0.6, n_order=10, subgraph_size=30, acc=0.69, epoch=2000, alpha=4, beta=0.01):
        super(FairGNN, self).__init__()

        parser = argparse.ArgumentParser()
        parser.add_argument('--no-cuda', action='store_true', default=False)
        parser.add_argument('--seed', type=int, default=1)
        parser.add_argument('--epochs', type=int, default=epoch)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--proj_hidden', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--sim_coeff', type=float, default=sim_coeff)
        parser.add_argument('--dataset', type=str, default='synthetic')
        parser.add_argument('--encoder', type=str, default='sage',
                            choices=['gcn', 'gin', 'sage', 'infomax', 'jk'])
        parser.add_argument('--batch_size', type=int, default=100)
        parser.add_argument('--subgraph_size', type=int, default=subgraph_size)
        parser.add_argument('--n_order', type=int, default=n_order)
        parser.add_argument('--hidden_size', type=int, default=1024)
        parser.add_argument('--experiment_type', type=str, default='train',
                            choices=['train', 'cf', 'test'])

        args = parser.parse_known_args()[0]
        args.num_hidden = 64
        args.alpha = alpha
        args.beta = beta
        args.acc = args.roc = acc

        nhid = args.num_hidden
        dropout = args.dropout
        self.estimator = GCN(nfeat, 1, dropout)
        self.GNN = get_model(nfeat, args)
        self.classifier = nn.Linear(nhid, 1)
        self.adv = nn.Linear(nhid, 1)

        G_params = (list(self.GNN.parameters())
                    + list(self.classifier.parameters())
                    + list(self.estimator.parameters()))
        self.optimizer_G = torch.optim.Adam(G_params, lr=args.lr,
                                            weight_decay=args.weight_decay)
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr=args.lr,
                                            weight_decay=args.weight_decay)

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()
        self.G_loss = 0
        self.A_loss = 0

    def fair_metric(self, sens, labels, output, idx):
        val_y = labels[idx].cpu().numpy()
        idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
        idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

        idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

        pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
        parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
        equality = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1))
        return parity, equality

    def fair_metric_direct(self, pred, labels, sens):
        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
        parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
        equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
        return parity.item(), equality.item()

    def forward(self, g, x):
        s = self.estimator(g, x)
        z = self.GNN(g, x)
        y = self.classifier(z)
        return y, s

    def optimize(self, g, x, labels, idx_train, sens, idx_sens_train, edge_index):
        self.train()

        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        s = self.estimator(edge_index, x)
        h = self.GNN(edge_index, x)
        y = self.classifier(h)
        s_g = self.adv(h)

        s_score = torch.sigmoid(s.detach())
        s_score[idx_sens_train] = sens[idx_sens_train].unsqueeze(1).float()
        y_score = torch.sigmoid(y)
        self.cov = torch.abs(
            torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))

        self.cls_loss = self.criterion(y[idx_train], labels[idx_train].unsqueeze(1).float())
        self.adv_loss = self.criterion(s_g, s_score)

        self.G_loss = self.cls_loss + self.args.alpha * self.cov - self.args.beta * self.adv_loss
        self.G_loss.backward()
        self.optimizer_G.step()

        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())
        self.A_loss = self.criterion(s_g, s_score)
        self.A_loss.backward()
        self.optimizer_A.step()

    def predict_sens_group(self, output, idx_test):
        """output: numpy array, idx_test: cpu tensor or numpy"""
        pred = output
        # idx_test를 numpy로 변환
        if torch.is_tensor(idx_test):
            idx_test_np = idx_test.cpu().numpy()
        else:
            idx_test_np = np.array(idx_test)

        sens_np = self.sens.detach().cpu().numpy()
        labels_np = self.labels.detach().cpu().numpy()

        result = []
        for s in [0, 1]:
            mask = (sens_np[idx_test_np] == s)
            F1 = f1_score(labels_np[idx_test_np][mask], pred[mask], average='micro')
            ACC = accuracy_score(labels_np[idx_test_np][mask], pred[mask])
            if self.labels.max() > 1:
                AUCROC = 0
            else:
                AUCROC = roc_auc_score(labels_np[idx_test_np][mask], pred[mask])
            result.extend([ACC, AUCROC, F1])
        return result

    def fit(self, g, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train, device='cuda'):
        self = self.to(device)
        features    = features.to(device)
        labels      = labels.to(device)
        idx_train   = idx_train.to(device)
        idx_val     = idx_val.to(device)
        sens        = sens.to(device)
        idx_sens_train = idx_sens_train.to(device)

        # idx_test는 CPU 유지 (numpy 인덱싱에 사용)
        idx_test_cpu = idx_test.cpu()

        args = self.args
        t_total = time.time()
        best_acc = 0
        self.temp_result     = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.temp_result_val = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        self.g      = g
        self.x      = features
        self.labels = labels
        self.sens   = sens

        self.edge_index = g.coalesce().indices().to(device)
        self.val_loss = 0

        for epoch in tqdm(range(args.epochs)):
            self.train()
            self.optimize(g, features, labels, idx_train, sens, idx_sens_train, self.edge_index)
            self.eval()
            output, s = self(self.edge_index, features)
            acc_val = accuracy(output[idx_val], labels[idx_val])

            if acc_val > args.acc or epoch == 0:
                if acc_val > best_acc:
                    best_epoch = epoch
                    best_acc = acc_val
                    self.val_loss = -acc_val.detach().cpu().item()
                    self.eval()
                    output, s = self.forward(self.edge_index, self.x)
                    output_np = (output > 0).long().detach().cpu().numpy()

                    labels_np = self.labels.detach().cpu().numpy()
                    sens_np   = self.sens.detach().cpu().numpy()
                    idx_np    = idx_test_cpu.numpy()

                    F1  = f1_score(labels_np[idx_np], output_np[idx_np], average='micro')
                    ACC = accuracy_score(labels_np[idx_np], output_np[idx_np])
                    AUCROC = 0 if self.labels.max() > 1 else \
                             roc_auc_score(labels_np[idx_np], output_np[idx_np])
                    ACC_sens0, AUCROC_sens0, F1_sens0, \
                    ACC_sens1, AUCROC_sens1, F1_sens1 = self.predict_sens_group(
                        output_np[idx_np], idx_test_cpu)
                    SP, EO = self.fair_metric_direct(
                        output_np[idx_np], labels_np[idx_np], sens_np[idx_np])

                    self.temp_result = (ACC, AUCROC, F1,
                                        ACC_sens0, AUCROC_sens0, F1_sens0,
                                        ACC_sens1, AUCROC_sens1, F1_sens1,
                                        SP, EO)

                    # val 결과
                    idx_val_cpu = idx_val.cpu()
                    idx_val_np  = idx_val_cpu.numpy()
                    F1  = f1_score(labels_np[idx_val_np], output_np[idx_val_np], average='micro')
                    ACC = accuracy_score(labels_np[idx_val_np], output_np[idx_val_np])
                    AUCROC = 0 if self.labels.max() > 1 else \
                             roc_auc_score(labels_np[idx_val_np], output_np[idx_val_np])
                    ACC_sens0, AUCROC_sens0, F1_sens0, \
                    ACC_sens1, AUCROC_sens1, F1_sens1 = self.predict_sens_group(
                        output_np[idx_val_np], idx_val_cpu)
                    SP, EO = self.fair_metric_direct(
                        output_np[idx_val_np], labels_np[idx_val_np], sens_np[idx_val_np])

                    self.temp_result_val = (ACC, AUCROC, F1,
                                            ACC_sens0, AUCROC_sens0, F1_sens0,
                                            ACC_sens1, AUCROC_sens1, F1_sens1,
                                            SP, EO)

            if epoch <= 10 and acc_val > args.acc:
                args.acc = acc_val

        print("Optimization Finished! Best Epoch:", best_epoch)
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    def predict(self, idx_test):
        return self.temp_result

    def predict_val(self, idx_val):
        return self.temp_result_val

    def predict_(self, idx_test):
        self.eval()
        output, s = self.forward(self.edge_index, self.x)
        output_np = (output > 0).long().detach().cpu().numpy()

        if torch.is_tensor(idx_test):
            idx_np = idx_test.cpu().numpy()
        else:
            idx_np = np.array(idx_test)

        labels_np = self.labels.detach().cpu().numpy()
        sens_np   = self.sens.detach().cpu().numpy()

        F1  = f1_score(labels_np[idx_np], output_np[idx_np], average='micro')
        ACC = accuracy_score(labels_np[idx_np], output_np[idx_np])
        AUCROC = 0 if self.labels.max() > 1 else \
                 roc_auc_score(labels_np[idx_np], output_np[idx_np])
        ACC_sens0, AUCROC_sens0, F1_sens0, \
        ACC_sens1, AUCROC_sens1, F1_sens1 = self.predict_sens_group(
            output_np[idx_np], torch.tensor(idx_np))
        SP, EO = self.fair_metric_direct(
            output_np[idx_np], labels_np[idx_np], sens_np[idx_np])

        return (ACC, AUCROC, F1,
                ACC_sens0, AUCROC_sens0, F1_sens0,
                ACC_sens1, AUCROC_sens1, F1_sens1,
                SP, EO)