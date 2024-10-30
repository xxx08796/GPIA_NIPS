from torch_geometric.data import Data
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
# import pulp
import gurobipy as gp

from utils.tools import truncate_value

EPS = 1e-5



def edit_function(graph, version="lp", num_candidate=10, num_perturb=5, ratio=0.05, is_normalize=True, num_layer=2):
    # if torch.cuda.is_available(): graph.to('cuda:0')
    value_list = []
    drop_total_edgelist = []
    drop_nodelist = []
    drop_edgelist = []
    train_idx = torch.where(graph.train_mask)[0]
    pos_idx = torch.where(graph.s == 1)[0]
    neg_idx = torch.where(graph.s == 0)[0]
    row, col, edge_attr = graph.adj_t.t().coo()
    edge_index = torch.stack([row, col], dim=0)
    for i in range(num_candidate):
        prop = truncate_value(i / (num_candidate - 1), EPS, 1 - EPS)
        weight = np.zeros(graph.num_nodes)
        weight[pos_idx], weight[neg_idx] = prop, 1 - prop
        prob = weight[train_idx]
        drop_node = np.random.choice(train_idx, size=int(graph.num_nodes * 0.05), replace=False, p=prob / np.sum(prob))
        # drop_node = np.random.choice(train_idx, size=int(graph.num_nodes * 0.05), replace=False)
        _, drop_edge1, _, _ = k_hop_subgraph(torch.tensor(drop_node), 1, edge_index, relabel_nodes=False)
        drop_nodelist.append(drop_node)

        unique_idx = torch.where(edge_index[0] < edge_index[1])[0]
        drop_idx = np.random.choice(unique_idx.cpu(), int(unique_idx.shape[0] * ratio), replace=False)
        drop_edge2 = edge_index[:, drop_idx]
        drop_edgelist.append(drop_idx)

        drop_edges = torch.cat((drop_edge1, drop_edge2), dim=1)
        unique_drop_edges = torch.unique(drop_edges, dim=1)
        drop_total_edgelist.append(unique_drop_edges)

        # unique_nodes = torch.unique(unique_drop_edges.flatten())
        influence_1, _, _, _ = k_hop_subgraph(torch.tensor(drop_node), num_layer, edge_index, relabel_nodes=False)
        influence_2, _, _, _ = k_hop_subgraph(torch.unique(drop_edge2.flatten()), num_layer-1, edge_index, relabel_nodes=False)

        # value_list.append(2 * len(unique_nodes) - len(drop_node))
        value_list.append(torch.unique(torch.cat((influence_1, influence_2), dim=0)).shape[0])

    value_list = np.array(value_list)

    distance_matrix = np.zeros((num_candidate, num_candidate))
    for i in range(num_candidate):
        for j in range(num_candidate):
            if i < j:
                # edit distance for edge
                edge_index1_tmp = drop_total_edgelist[i].numpy()
                hash_set1 = set(hash(tuple(edge)) for edge in zip(edge_index1_tmp[0], edge_index1_tmp[1]))
                edge_index2_tmp = drop_total_edgelist[j].numpy()
                hash_set2 = set(hash(tuple(edge)) for edge in zip(edge_index2_tmp[0], edge_index2_tmp[1]))
                duplicate_hashes = hash_set1.intersection(hash_set2)
                duplicate_edges = [tuple(edge) for edge in zip(edge_index1_tmp[0], edge_index1_tmp[1]) if
                                   hash(tuple(edge)) in duplicate_hashes]
                edge_distance = edge_index1_tmp.shape[1] + edge_index2_tmp.shape[1] - 2 * len(duplicate_edges)

                # edit distance for node
                node_index1_tmp = set(drop_nodelist[i])
                node_index2_tmp = set(drop_nodelist[j])
                intersection = set(node_index1_tmp) & set(node_index2_tmp)
                node_distance = len(node_index1_tmp) + len(node_index2_tmp) - 2 * len(intersection)
                distance_matrix[i][j] = edge_distance + node_distance
                distance_matrix[j][i] = edge_distance + node_distance
            elif i == j:
                distance_matrix[i][j] = 0

    # sns.heatmap(distance_matrix, cmap='coolwarm', annot=True, fmt=".2f")
    # plt.show()

    if is_normalize is True:
        distance_matrix = distance_matrix / np.linalg.norm(distance_matrix, axis=1, keepdims=True)
        value_list = value_list / np.linalg.norm(value_list, axis=0, keepdims=True)

    def max_submatrix_sum(value_list, matrix):
        max_sum = float('-inf')
        max_submatrix = None
        max_i = 0

        for i in tqdm(range(100)):
            random_idx = np.random.choice(np.arange(0, num_candidate), size=num_perturb, replace=False)
            new_matrix = matrix[random_idx][:, random_idx]
            submatrix_sum = np.sum(new_matrix)
            value_sum = -np.sum(value_list[random_idx])
            if submatrix_sum > max_sum:
                max_sum = submatrix_sum + value_sum
                max_submatrix = new_matrix
                max_i = random_idx

        return max_sum, max_submatrix, max_i

    def max_submatrix_sum_pulp(value_list, matrix):
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        selected = [model.addVar(vtype=gp.GRB.BINARY, name=f"x{i}") for i in range(len(value_list))]

        diverse, error = 0, 0
        sum_value = 0
        for i in range(len(selected)):
            error += (selected[i] * value_list[i])
            sum_value = selected[i] + sum_value
            for j in range(len(selected)):
                diverse += (selected[i] * selected[j] * matrix[i][j])
        model.setObjective(diverse, sense=gp.GRB.MAXIMIZE)
        model.addConstr(error <= float('inf'))
        model.addConstr(sum_value == num_perturb)
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            idx = []
            for i in range(len(selected)):
                if selected[i].x > 0:
                    idx.append(i)
        else:
            print("Optimization failed. Status:", model.status)
        return idx

    if version == 'lp':
        idx = max_submatrix_sum_pulp(value_list, distance_matrix)
    else:
        _, _, idx = max_submatrix_sum(value_list, distance_matrix)
    drop_edge_idx = [drop_edgelist[k] for k in idx]
    drop_node_idx = [drop_nodelist[k] for k in idx]
    return drop_edge_idx, drop_node_idx


def edit_function_intra(graph, version="lp", num_candidate=10, num_perturb=5, ratio=0.05, is_normalize=True, num_layer=2):
    # if torch.cuda.is_available(): graph.to('cuda:0')
    value_list = []
    drop_total_edgelist = []
    drop_nodelist = []
    drop_edgelist = []
    train_idx = torch.where(graph.train_mask)[0]
    pos_idx = torch.where(graph.s == 1)[0]
    neg_idx = torch.where(graph.s == 0)[0]
    row, col, edge_attr = graph.adj_t.t().coo()
    edge_index = torch.stack([row, col], dim=0)
    for i in range(num_candidate):
        prop = truncate_value(i / (num_candidate - 1), EPS, 1 - EPS)
        weight = np.zeros(graph.num_nodes)
        weight[pos_idx], weight[neg_idx] = prop, 1 - prop
        prob = weight[train_idx]
        drop_node = np.random.choice(train_idx, size=int(graph.num_nodes * ratio), replace=False, p=prob / np.sum(prob))
        # drop_node = np.random.choice(train_idx, size=int(graph.num_nodes * 0.05), replace=False)
        _, drop_edge1, _, _ = k_hop_subgraph(torch.tensor(drop_node), 1, edge_index, relabel_nodes=False, num_nodes=graph.num_nodes)
        drop_nodelist.append(drop_node)

        unique_idx = torch.where(edge_index[0] < edge_index[1])[0]
        drop_idx = np.random.choice(unique_idx.cpu(), int(unique_idx.shape[0] * ratio), replace=False)
        drop_edge2 = edge_index[:, drop_idx]
        drop_edgelist.append(drop_idx)

        drop_edges = torch.cat((drop_edge1, drop_edge2), dim=1)
        unique_drop_edges = torch.unique(drop_edges, dim=1)
        drop_total_edgelist.append(unique_drop_edges)

        # unique_nodes = torch.unique(unique_drop_edges.flatten())
        influence_1, _, _, _ = k_hop_subgraph(torch.tensor(drop_node), num_layer, edge_index, relabel_nodes=False, num_nodes=graph.num_nodes)
        influence_2, _, _, _ = k_hop_subgraph(torch.unique(drop_edge2.flatten()), num_layer-1, edge_index, relabel_nodes=False, num_nodes=graph.num_nodes)

        # value_list.append(2 * len(unique_nodes) - len(drop_node))
        value_list.append(torch.unique(torch.cat((influence_1, influence_2), dim=0)).shape[0])

    value_list = np.array(value_list)

    distance_matrix = np.zeros((num_candidate, num_candidate))
    for i in range(num_candidate):
        for j in range(num_candidate):
            if i < j:
                # edit distance for edge
                edge_index1_tmp = drop_total_edgelist[i].numpy()
                hash_set1 = set(hash(tuple(edge)) for edge in zip(edge_index1_tmp[0], edge_index1_tmp[1]))
                edge_index2_tmp = drop_total_edgelist[j].numpy()
                hash_set2 = set(hash(tuple(edge)) for edge in zip(edge_index2_tmp[0], edge_index2_tmp[1]))
                duplicate_hashes = hash_set1.intersection(hash_set2)
                duplicate_edges = [tuple(edge) for edge in zip(edge_index1_tmp[0], edge_index1_tmp[1]) if
                                   hash(tuple(edge)) in duplicate_hashes]
                edge_distance = edge_index1_tmp.shape[1] + edge_index2_tmp.shape[1] - 2 * len(duplicate_edges)

                # edit distance for node
                node_index1_tmp = set(drop_nodelist[i])
                node_index2_tmp = set(drop_nodelist[j])
                intersection = set(node_index1_tmp) & set(node_index2_tmp)
                node_distance = len(node_index1_tmp) + len(node_index2_tmp) - 2 * len(intersection)
                distance_matrix[i][j] = edge_distance + node_distance
                distance_matrix[j][i] = edge_distance + node_distance
            elif i == j:
                distance_matrix[i][j] = 0

    # sns.heatmap(distance_matrix, cmap='coolwarm', annot=True, fmt=".2f")
    # plt.show()

    if is_normalize is True:
        distance_matrix = distance_matrix / np.linalg.norm(distance_matrix, axis=1, keepdims=True)
        value_list = value_list / np.linalg.norm(value_list, axis=0, keepdims=True)

    def max_submatrix_sum(value_list, matrix):
        max_sum = float('-inf')
        max_submatrix = None
        max_i = 0

        for i in tqdm(range(100)):
            random_idx = np.random.choice(np.arange(0, num_candidate), size=num_perturb, replace=False)
            new_matrix = matrix[random_idx][:, random_idx]
            submatrix_sum = np.sum(new_matrix)
            value_sum = -np.sum(value_list[random_idx])
            if submatrix_sum > max_sum:
                max_sum = submatrix_sum + value_sum
                max_submatrix = new_matrix
                max_i = random_idx

        return max_sum, max_submatrix, max_i

    def max_submatrix_sum_pulp(value_list, matrix):
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        selected = [model.addVar(vtype=gp.GRB.BINARY, name=f"x{i}") for i in range(len(value_list))]

        diverse, error = 0, 0
        sum_value = 0
        for i in range(len(selected)):
            error += (selected[i] * value_list[i])
            sum_value = selected[i] + sum_value
            for j in range(len(selected)):
                diverse += (selected[i] * selected[j] * matrix[i][j])
        model.setObjective(diverse, sense=gp.GRB.MAXIMIZE)
        model.addConstr(error <= float('inf'))
        model.addConstr(sum_value == num_perturb)
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            idx = []
            for i in range(len(selected)):
                if selected[i].x > 0:
                    idx.append(i)
        else:
            print("Optimization failed. Status:", model.status)
        return idx

    if version == 'lp':
        idx = max_submatrix_sum_pulp(value_list, distance_matrix)
    else:
        _, _, idx = max_submatrix_sum(value_list, distance_matrix)
    drop_edge_idx = [drop_edgelist[k] for k in idx]
    drop_node_idx = [drop_nodelist[k] for k in idx]
    return drop_edge_idx, drop_node_idx
