from torch.utils.data import DataLoader, random_split
from an_expe.model import OrderPredictor
from an_expe.dataset import AdjNounDataset

import fasttext
import json
import copy
import argparse
import torch
from tqdm import tqdm
import sys
import copy

epsi = sys.float_info.epsilon


def test_model(model, val_dataloader, verbose=True):
    model.eval()
    a_0 = 0
    a_1 = 0
    a_tot = 0
    for i, batch in enumerate(val_dataloader):
        x, y = batch
        y_hat = model(x)
        y = y.unsqueeze(-1)
        tp = ((y_hat >= 0) & (y == 1)).sum()
        a_1 += tp / (((y == 1).sum()) + epsi)
        tp_ = ((y_hat < 0) & (y == 0)).sum()
        a_0 += tp_ / (((y == 0).sum()) + epsi)
        a_tot += (tp + tp_) / y.shape[0]
    return float(a_0 / len(val_dataloader)), float(
        a_1 / len(val_dataloader)), float(a_tot / len(val_dataloader))


def tq(iterator, verbose, **args):
    if verbose:
        return tqdm(iterator, **args)
    return iterator


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def main(config_file, weights):
    config = load_config(config_file)
    print("Configuration")
    print(config)
    print("-" * 60)
    torch.manual_seed(config['seed'])

    ds = AdjNounDataset(config['dataset_path'])

    split = config['split']
    _, _, test_ds = random_split(ds, split)

    op = OrderPredictor(embeddingModel=fasttext.load_model(
        config['embeddingModel']),
                        num_layers=config['num_layers'],
                        hidden_dim=config['hidden_dim'],
                        dropout=config['dropout'])

    test_dataloader = DataLoader(test_ds, batch_size=config['batch_size'])

    op.load_state_dict(torch.load(weights))
    best_model = op
    best_model.eval()
    test_acc = test_model(best_model, test_dataloader)
    print("Full model")
    print(
        f"Test Accuracy: an : {test_acc[0]:.2f}, na : {test_acc[1]:.2f}, total : {test_acc[2]:.2f}"
    )

    only_adj = copy.copy(best_model)
    with torch.no_grad():
        only_adj.batch_norm.weight[0] = 0

    test_acc = test_model(only_adj, test_dataloader)
    print("Model without noun activation")
    print(
        f"Test Accuracy: an : {test_acc[0]:.2f}, na : {test_acc[1]:.2f}, total : {test_acc[2]:.2f}"
    )
    only_noun = copy.copy(best_model)
    with torch.no_grad():
        only_noun.batch_norm.weight[1] = 0

    test_acc = test_model(only_noun, test_dataloader)
    print("Model without adj activation")
    print(
        f"Test Accuracy: an : {test_acc[0]:.2f}, na : {test_acc[1]:.2f}, total : {test_acc[2]:.2f}"
    )
if __name__ == "__main__":
    # Argument parser to optionally specify the config file path
    parser = argparse.ArgumentParser(
        description="Run OrderPredictor Training Script")
    parser.add_argument(
        '--config',
        type=str,
        default="config/an_expe/config.json",
        help=
        "Path to the configuration file. Default is 'config/an_expe/config.json'."
    )
    parser.add_argument(
        '--weights',
        type=str,
        default="models/best.pt",
        help=
        "Path to the weights of the trained model. Default is 'models/best.pt'"
    )
    args = parser.parse_args()

    main(args.config, args.weights)
