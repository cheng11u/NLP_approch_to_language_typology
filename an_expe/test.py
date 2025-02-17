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

epsi = sys.float_info.epsilon


def validate_model(model, val_dataloader, verbose=True):
    model.eval()
    a_0 = 0
    a_1 = 0
    a_tot = 0
    na_total = an_total =  0
    for i, batch in enumerate(val_dataloader):
        #print(batch)
        x, y = batch
        y_hat = model(x)
        #print(y_hat)
        na_total += y.sum()
        an_total += (y==0).sum()
        y = y.unsqueeze(-1)
        tp = ((y_hat >= 0) & (y == 1)).sum()
        a_1 += tp / (((y == 1).sum()) + epsi)
        tp_ = ((y_hat < 0) & (y == 0)).sum()
        a_0 += tp_ / (((y == 0).sum()) + epsi)
        a_tot += (tp + tp_) / y.shape[0]
        bad_indexes = ([i for i, u in enumerate((y_hat >= 0) != (y == 1)) if u])
        if verbose:
            print("Errors", [x['A'][i] + "|" + x['N'][i] + str(int(y[i,0])) for i in bad_indexes])
    if verbose:
        print("Number of NA :", na_total.numpy(),"Number of AN :",an_total.numpy())
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


def main(config_file):
    config = load_config(config_file)
    print("Configuration")
    print(config)
    print("-" * 60)
    torch.manual_seed(config['seed'])

    ds = AdjNounDataset(config['dataset_path'])

    split = config['split']

    op = OrderPredictor(embeddingModel=fasttext.load_model(
        config['embeddingModel']),
                        num_layers=config['num_layers'],
                        hidden_dim=config['hidden_dim'],
                        dropout=config['dropout'])

    test_dataloader = DataLoader(ds, batch_size=config['batch_size'])

    op.load_state_dict(torch.load("models/best.pt"))
    best_model = op
    test_acc = validate_model(best_model, test_dataloader)
    print(
        f"Test Accuracy: an : {test_acc[0]:.2f}, na : {test_acc[1]:.2f}, total : {test_acc[2]:.2f}"
    )

    print("Weights")
    print(best_model.batch_norm.weight)
    print("Biases")
    print(best_model.batch_norm.bias)


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

    args = parser.parse_args()

    # Use the provided config file or fallback to default
    main(args.config)
