import time
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch_geometric.nn import GraphSAGE
from utils.tools import EarlyStopper
torch.cuda.empty_cache()
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
from models.gnn_base import GNNBase


class NodeClassifier(GNNBase):
    def __init__(self, args, n_features, n_classes, ):
        super(NodeClassifier, self).__init__()
        self.args = args
        self.target_model = args.target_model
        self.device = torch.device(args.device)
        self.model = self.determine_model(n_features, n_classes, args).to(self.device)
        self.ref_model_param = None

    def determine_model(self, num_features, num_classes, args):
        if self.target_model == 'SAGE':
            self.lr, self.decay = 1e-4, 5e-4
            return GraphSAGE(in_channels=num_features, hidden_channels=64, out_channels=num_classes, num_layers=2)
        else:
            raise NotImplementedError

    def train_model(self, data, unlearn_info=None, trained=False, save_flag=False, times=None):
        self.model.train()
        self.model.reset_parameters()
        self.model, data = self.model.to(self.device), data.to(self.device)
        early_stopper = EarlyStopper(patience=50, min_delta=0)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        if not trained:
            train_time = time.time()
            for epoch in range(self.args.num_epochs):
                optimizer.zero_grad()
                out = self.model(data.x, data.adj_t)
                train_loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], reduction='sum')
                if self.args.early_stop:
                    val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask], reduction='sum')
                    if early_stopper.early_stop(val_loss):
                        print("early stop at epoch {}".format(epoch))
                        break
                train_loss.backward()
                optimizer.step()
            res = self.evaluate(data)
            if save_flag is True:
                self.ref_model_param = self.model.state_dict()
            # times['train_time_l'].append(time.time() - train_time)

        else:
            # state_dict = torch.load(trained_path)
            self.model.load_state_dict(self.ref_model_param)
            res = self.evaluate(data)
        if unlearn_info is None:
            return res, None

        out1 = self.model(data.x, data.adj_t)
        # out3 = self.model(data.x, data.edge_index)
        out2 = self.model(data.x_unlearn, data.adj_t_unlearn)
        if self.args.unlearn_task == "node_with_edge":
            mask1 = np.array([False] * out1.shape[0])
            mask1[unlearn_info[0]] = True
            mask1[unlearn_info[2]] = True
            mask2 = np.array([False] * out2.shape[0])
            mask2[unlearn_info[2]] = True
        elif self.args.unlearn_task == "node":
            mask1 = np.array([False] * out1.shape[0])
            mask1[unlearn_info[0]] = True
            mask1[unlearn_info[2]] = True
            mask2 = np.array([False] * out2.shape[0])
            mask2[unlearn_info[2]] = True
        else:
            raise NotImplementedError

        loss1 = F.cross_entropy(out1[mask1], data.y[mask1], reduction='sum')
        loss2 = F.cross_entropy(out2[mask2], data.y[mask2], reduction='sum')
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        loss_p = loss1 - loss2
        vs = grad(loss_p, model_params, retain_graph=True, create_graph=True)
        return res, vs

    def evaluate_unlearn(self, new_parameters, data):
        idx = 0
        for p in self.model.parameters():
            p.data = new_parameters[idx]
            idx = idx + 1

        with torch.no_grad():
            self.model.eval()
            out = self.model(data.x_unlearn, data.adj_t_unlearn)
            out = F.softmax(out, dim=-1)
            if data.num_classes == 2:
                test_res = roc_auc_score(data.y[data.test_mask].cpu(), out[data.test_mask, 1].cpu())
                train_res = roc_auc_score(data.y[data.retrain_mask].cpu(), out[data.retrain_mask, 1].cpu())
            else:
                pred_s = np.argmax(out.cpu().numpy(), axis=1)
                test_res = accuracy_score(pred_s[data.test_mask.cpu()], data.y[data.test_mask].cpu().numpy())
                train_res = accuracy_score(pred_s[data.retrain_mask.cpu()], data.y[data.retrain_mask].cpu().numpy())
        return {'train_res': train_res, 'test_res': test_res}

    @torch.no_grad()
    def evaluate(self, data):
        self.model.eval()
        out = self.model(data.x, data.adj_t)
        out = F.softmax(out, dim=-1)
        if data.num_classes == 2:
            test_res = roc_auc_score(data.y[data.test_mask].cpu(), out[data.test_mask, 1].cpu())
            train_res = roc_auc_score(data.y[data.train_mask].cpu(), out[data.train_mask, 1].cpu())
        else:
            pred_s = np.argmax(out.cpu().numpy(), axis=1)
            test_res = accuracy_score(pred_s[data.test_mask.cpu()], data.y[data.test_mask].cpu().numpy())
            train_res = accuracy_score(pred_s[data.train_mask.cpu()], data.y[data.train_mask].cpu().numpy())
        return {'train_res': train_res, 'test_res': test_res}

    @torch.no_grad()
    def posterior(self, x, adj_t):
        """
        Returns: posterior of all node in data
        """
        self.model.eval()
        self.model, x, adj_t = self.model.to(self.device), x.to(self.device), adj_t.to(self.device)
        # self._gen_test_loader()
        out = self.model(x, adj_t)
        # for _, mask in data('test_mask'):
        #     posteriors = F.log_softmax(posteriors[mask], dim=-1)
        posteriors = F.softmax(out, dim=-1)
        return posteriors
