from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import Tensor
from torch.nn.utils import vector_to_parameters
from torch_geometric.nn import GraphSAGE
from models.deepset import DeepSets
from utils.tools import  map_to_label



def repackage_graphsage_parameters(model, model_name='SAGE'):
    repackaged_params = []
    if model_name == 'SAGE':
        for layer in model.convs:
            weights = layer.lin_l.weight
            biases = layer.lin_l.bias
            repackaged_params.append([weights.detach().numpy(), biases.view(-1, 1).detach().numpy()])
            # weights = layer.lin_r.weight
            # repackaged_params.append(weights.detach().numpy())
    else:
        raise NotImplementedError

    return repackaged_params


def whitebox_deepsets_attack(opt, exp_idx, x_train: Tensor, r_train: Tensor):
    if opt.data == 'facebook':
        num_features, num_label_classes = 1282, 2
    elif opt.data == 'pubmed':
        num_features, num_label_classes = 500, 3
    elif opt.data == 'pokec':
        num_features, num_label_classes = 197, 2
    else:
        raise NotImplementedError

    test_output_dir = f'./data/{opt.data}_output_target/'
    x_test = torch.load(test_output_dir + 'parameter_test_300_early_stop_True.pt')
    r_test = torch.load(test_output_dir + 'label_test_300_early_stop_True.pt').cpu()
    shadow_labels, target_labels = map_to_label(r_train, opt.num_class), map_to_label(r_test, opt.num_class)

    targets = []
    for params_vector in x_test:
        model = GraphSAGE(
            in_channels=num_features,
            hidden_channels=64,
            out_channels=num_label_classes,
            num_layers=2,
        )
        model = model.to(params_vector.device)
        vector_to_parameters(params_vector, model.parameters())
        targets.append(model)
    shadow_models = []
    for params_vector in x_train:
        model = GraphSAGE(
            in_channels=num_features,
            hidden_channels=64,
            out_channels=num_label_classes,
            num_layers=2,
        )
        model = model.to(params_vector.device)
        vector_to_parameters(params_vector, model.parameters())
        shadow_models.append(model)
    hyperparams = dict()

    latent_dim = opt.atk_num_hidden
    epochs = opt.atk_epochs
    lr = opt.atk_lr
    wd = opt.atk_wd
    out_dim = 1

    meta_classifier = DeepSets(repackage_graphsage_parameters(shadow_models[0]), latent_dim=latent_dim,
                               epochs=epochs, lr=lr, wd=wd, n_classes=opt.num_class, use_weight=False)

    train = [repackage_graphsage_parameters(s) for s in shadow_models]
    test = [repackage_graphsage_parameters(t) for t in targets]

    meta_classifier.fit(train, shadow_labels)
    prob = meta_classifier.predict(test)
    prob = F.softmax(prob, dim=-1)
    y_pred = torch.argmax(prob, dim=-1)
    acc = accuracy_score(target_labels, y_pred)
    auc = roc_auc_score(target_labels, prob[:, 1])
    print(f'Attack acc: {acc}, roc_auc: {auc}')
    return acc, auc