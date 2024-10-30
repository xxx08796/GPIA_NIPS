import argparse
import random
import time
import numpy as np
import torch
from dgl.data import register_data_args
from torch.nn.utils import parameters_to_vector
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.utils import k_hop_subgraph
from torch_sparse import SparseTensor
from models.node_classifier import NodeClassifier
from utils.edit_func import edit_function
from utils.cg import my_cg
from utils.tools import truncate_value, ExperimentMetrics, data_split
import torch_geometric.transforms as T
from wihtebox_attack import whitebox_deepsets_attack

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

EPS = 1e-5


class GraphInfluence:
    def __init__(self, opt, data):
        super(GraphInfluence, self).__init__()
        self.edge_index_nu = None
        self.opt = opt
        self.remain_nodes = None
        self.deleted_nodes, self.feature_nodes, self.influence_nodes = None, None, None
        self.data = data
        self.target_model = NodeClassifier(self.opt, self.data.num_features, self.data.num_classes)

    def unlearn(self, drop_nodes, drop_edge_unique_idx, trained):
        self.unlearn_request(drop_nodes, drop_edge_unique_idx)
        train_res, vs = self.train_model(trained)
        unlearn_res = self.gif_approx(vs)
        print('unlearn result: ', unlearn_res)
        unlearn_param = self.get_param().cpu().detach()
        return self.get_new_r(), unlearn_res, unlearn_param

    def get_post(self):
        post = self.target_model.posterior(self.data.x_unlearn, self.data.adj_t_unlearn)
        return post[self.remain_nodes, :]

    def get_param(self):
        param = parameters_to_vector(self.target_model.model.parameters())
        return param

    def get_new_r(self):
        new_s = self.data.s[self.remain_nodes]
        new_r = torch.sum(new_s) / new_s.shape[0]
        # new_p = 0 if new_r < 0.5 else 1
        return new_r.cpu()

    def unlearn_request(self, drop_nodes, drop_edge_unique_idx):
        self.remain_nodes = np.setdiff1d(np.arange(self.data.num_nodes), drop_nodes)
        self.data.retrain_mask = self.data.train_mask.clone()
        self.data.retrain_mask[drop_nodes] = False
        self.data.x_unlearn = self.data.x.clone()
        row, col, edge_attr = self.data.adj_t.t().coo()
        self.edge_index_nu = torch.stack([row, col], dim=0)
        if self.opt.unlearn_task == 'node':
            edge_index_unlearn, _ = subgraph(
                torch.tensor(self.remain_nodes, device=self.edge_index_nu.device),
                self.edge_index_nu,
                num_nodes=self.data.num_nodes,
                relabel_nodes=False
            )
            self.data.adj_t_unlearn = SparseTensor(
                row=edge_index_unlearn[0],
                col=edge_index_unlearn[1],
                sparse_sizes=(self.data.num_nodes, self.data.num_nodes)
            ).to(self.edge_index_nu.device)
            self.find_k_hops(drop_nodes)

        if self.opt.unlearn_task == 'node_with_edge':
            # assert is_undirected(edge_index_tmp, num_nodes=self.data.num_nodes)
            drop_edges = self.edge_index_nu[:, drop_edge_unique_idx]
            unique_idx = torch.where(self.edge_index_nu[0] < self.edge_index_nu[1])[0].cpu()
            remain_idx = np.setdiff1d(unique_idx, drop_edge_unique_idx)
            edge_index_tmp = self.edge_index_nu[:, remain_idx]
            edge_index_tmp = to_undirected(edge_index_tmp, num_nodes=self.data.num_nodes)
            edge_index_unlearn, _ = subgraph(
                torch.tensor(self.remain_nodes, device=edge_index_tmp.device),
                edge_index_tmp,
                num_nodes=self.data.num_nodes,
                relabel_nodes=False
            )
            self.data.adj_t_unlearn = SparseTensor(
                row=edge_index_unlearn[0],
                col=edge_index_unlearn[1],
                sparse_sizes=(self.data.num_nodes, self.data.num_nodes)
            ).to(self.edge_index_nu.device)
            self.find_k_hops(drop_nodes, drop_edges)

        # if self.opt.unlearn_task == 'feature':
        #     unique_nodes = np.random.choice(len(self.train_indices),
        #                                     int(len(self.train_indices) * self.opt.unlearn_ratio),
        #                                     replace=False)
        #     self.data.x_unlearn[unique_nodes] = 0.

    def train_model(self, trained):
        res, vs = self.target_model.train_model(
            self.data,
            (self.deleted_nodes, self.feature_nodes, self.influence_nodes),
            trained,
            save_flag=True
        )
        return res, vs

    def retrain_model(self):
        new_data = Data(
            x=self.data.x_unlearn,
            adj_t=self.data.adj_t_unlearn,
            y=self.data.y,
            train_mask=self.data.retrain_mask,
            test_mask=self.data.test_mask,
            num_classes=self.data.num_classes
        )
        self.target_model.model.reset_parameters()
        res, _ = self.target_model.train_model(new_data)
        return res

    def find_k_hops(self, drop_node, drop_edge=None):
        hops = args.num_layers
        if self.opt.unlearn_task == 'node':
            subset, _, _, _ = k_hop_subgraph(torch.tensor(drop_node), hops,
                                             self.edge_index_nu, relabel_nodes=False)
            neighbor_nodes = np.setdiff1d(subset.cpu(), drop_node)
            self.deleted_nodes = drop_node
            non_missing = torch.where(self.data.y > -1)[0]
            self.influence_nodes = np.intersect1d(neighbor_nodes, non_missing.cpu())

        # if self.opt.unlearn_task == 'feature':
        #     self.feature_nodes = unique_nodes
        #     self.influence_nodes = neighbor_nodes

        if self.opt.unlearn_task == 'node_with_edge':
            subset, _, _, _ = k_hop_subgraph(torch.tensor(drop_node), hops,
                                             self.edge_index_nu, relabel_nodes=False)
            neighbor_nodes = np.setdiff1d(subset.cpu(), drop_node)
            self.deleted_nodes = drop_node
            # non_missing = torch.where(self.data.y > -1)[0]
            self.influence_nodes = neighbor_nodes
            drop_node_edge = np.unique(drop_edge.cpu())
            subset, _, _, _ = k_hop_subgraph(torch.tensor(drop_node_edge), hops - 1,
                                             self.edge_index_nu, relabel_nodes=False)
            self.influence_nodes = np.union1d(self.influence_nodes, subset.cpu())

    def gif_approx(self, deltas):
        model_params = [p for p in self.target_model.model.parameters() if p.requires_grad]
        influence, loss, status = my_cg(self.data, self.target_model.model, self.data.edge_index, deltas, self.opt.damp,
                                        self.opt.device)
        params_est = [p1 + p2 for p1, p2 in zip(influence, model_params)]
        unlearn_res = self.target_model.evaluate_unlearn(params_est, self.data)
        return unlearn_res


def gnn_approx(opt, data: Data, res):
    data = T.ToSparseTensor()(data)
    data.train_mask, data.val_mask, data.test_mask = data_split(data.y, train_ratio=0.7, val_ratio=0.1)
    classes = set(data.y.numpy())
    classes.discard(-1)
    data.num_classes = len(classes)
    graph_influ = GraphInfluence(opt, data)
    drop_edge_idx_list, drop_node_idx_list = edit_function(data, num_candidate=args.num_candidate,
                                                           num_perturb=args.num_shadow)
    train_idx = torch.where(data.train_mask)[0]
    pos_idx = torch.where(data.s == 1)[0]
    neg_idx = torch.where(data.s == 0)[0]
    for idx in range(opt.num_shadow):
        print(idx)
        trained = False if idx == 0 else True
        prop = truncate_value(idx / (opt.num_shadow - 1), EPS, 1 - EPS)
        weight = np.zeros(data.num_nodes)
        weight[pos_idx], weight[neg_idx] = prop, 1 - prop
        prob = weight[train_idx]
        drop_idx = np.random.choice(train_idx, opt.num_drop, replace=False, p=prob / np.sum(prob))
        ratio, unlearn_res, unlearn_param = graph_influ.unlearn(drop_node_idx_list[idx],
                                                                drop_edge_idx_list[idx], trained)
        res['unlearn_param_train'].append(unlearn_param)
        res['r_train'].append(ratio)
        res['unlearn_train_res'].append(unlearn_res['train_res'])
        res['unlearn_test_res'].append(unlearn_res['test_res'])
    return


def main():
    result = {'unlearn_train_res': [], 'unlearn_test_res': [],
              'unlearn_param_train': [], 'r_train': []}
    train_idx = np.arange(args.num_train)
    np.random.shuffle(train_idx)
    start_time = time.time()

    for j, idx in enumerate(train_idx):
        print(f'========reference graph {j}========')
        ref_graph = torch.load('{}/train_shadow_idx_{}.pt'.format(ref_graph_dir, idx))
        gnn_approx(args, ref_graph, result)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('train time: ', elapsed_time)
    test_res = np.array(result['unlearn_test_res'])
    train_res = np.array(result['unlearn_train_res'])
    print('train_diff:', np.mean(train_res), np.std(train_res))
    print('test_diff:', np.mean(test_res), np.std(test_res))
    unlearn_param_train = torch.vstack(result['unlearn_param_train'])
    # output_dir = f'./data/{args.data}_unlearn_output/'
    r_train = torch.stack(result['r_train'])
    # torch.save(unlearn_param_train,
    #            output_dir + f'parameter_num_train_{args.num_train}_num_shadow_{args.num_shadow}.pt')
    # torch.save(r_train, output_dir + f'label_num_train_{args.num_train}_num_shadow_{args.num_shadow}.pt')

    return unlearn_param_train, r_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1500)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_shadow", type=int, default=4)
    parser.add_argument("--num_candidate", type=int, default=8)
    parser.add_argument("--num_drop", type=int, default=25)
    parser.add_argument("--num_train", type=int, default=50)
    parser.add_argument("--num_test", type=int, default=300)
    parser.add_argument('--num_exp', type=int, default=5)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--data', type=str, default='facebook')
    parser.add_argument('--target_model', type=str, default='SAGE')
    parser.add_argument('--unlearn_task', type=str, default='node_with_edge')
    parser.add_argument("--atk_num_hidden", type=int, default=128)
    parser.add_argument("--atk_epochs", type=int, default=200)
    parser.add_argument("--atk_lr", type=float, default=1e-3)
    parser.add_argument("--atk_wd", type=float, default=5e-4)
    parser.add_argument('--damp', type=float, default=50)
    parser.add_argument('--aggr', type=str, default='diff')
    parser.add_argument('--cg_method', type=str, default='my_cg')
    parser.add_argument('--sample_method', type=str, default='comm_prob')
    parser.add_argument('--jump_out', type=float, default=0.4)
    args = parser.parse_args()
    print(args)
    ref_graph_dir = f'./data/{args.data}_graph_unlearn/train_{args.num_train}'
    metrics = ExperimentMetrics()
    param_train, r_train = main()

    for i in range(args.num_exp):
        acc, auc = whitebox_deepsets_attack(args, i, param_train, r_train)
        metrics.add_metrics(acc, auc)
    metrics.calculate_statistics()
