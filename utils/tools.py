from sklearn.model_selection import train_test_split
from torch_geometric.utils import index_to_mask
import numpy as np
import torch


class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def truncate_value(value, min_value, max_value):
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    else:
        return value


def map_to_label(value, num_class):
    value = np.array(value)
    value = np.minimum(1.0, np.maximum(0.0, value))
    label = (value * num_class).astype(int)
    label[value == 1] = num_class - 1
    return label


class ExperimentMetrics:
    def __init__(self):
        self.accuracy = []
        self.auc = []

    def add_metrics(self, acc, auc):
        self.accuracy.append(acc)
        self.auc.append(auc)

    def calculate_statistics(self):
        for name, values in zip(
                ["accuracy",  "auc"],
                [self.accuracy,  self.auc]
        ):
            mean = np.mean(values)
            stdev = np.std(values)
            print(f"{name.capitalize()}: {mean:.4f}+{stdev:.4f}")


def data_split(label, train_ratio, val_ratio):
    # train val test split
    non_missing_idx = np.where(label.numpy() >= 0)[0]
    non_missing_label = label.numpy()[non_missing_idx]
    x_remaining, x_test, y_remaining, y_test = train_test_split(non_missing_idx, non_missing_label,
                                                                test_size=1 - train_ratio - val_ratio,
                                                                stratify=non_missing_label, random_state=42)
    if val_ratio == 0:
        train_mask = index_to_mask(torch.LongTensor(x_remaining), size=label.shape[0])
        test_mask = index_to_mask(torch.LongTensor(x_test), size=label.shape[0])
        return train_mask, None, test_mask

    # Adjusts val ratio, w.r.t. remaining dataset.
    remaining_ratio = train_ratio + val_ratio
    ratio_val_adjusted = val_ratio / remaining_ratio
    # Produces train and val splits.
    x_train, x_val, y_train, y_val = train_test_split(x_remaining, y_remaining,
                                                      test_size=ratio_val_adjusted,
                                                      stratify=y_remaining, random_state=42)
    train_mask = index_to_mask(torch.LongTensor(x_train), size=label.shape[0])
    val_mask = index_to_mask(torch.LongTensor(x_val), size=label.shape[0])
    test_mask = index_to_mask(torch.LongTensor(x_test), size=label.shape[0])
    return train_mask, val_mask, test_mask