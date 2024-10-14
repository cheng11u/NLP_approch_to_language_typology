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
    train_ds, val_ds, test_ds = random_split(ds, split)

    op = OrderPredictor(embeddingModel=fasttext.load_model(
        config['embeddingModel']),
                        num_layers=config['num_layers'],
                        hidden_dim=config['hidden_dim'],
                        dropout=config['dropout'])
    op.eval()
    op.load_state_dict(torch.load("models/best.pt"))

    print(op({"A": ["beau", "gentil"], "N": ["chien","arbre"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OrderPredictor compare script")
    parser.add_argument(
        '--config',
        type=str,
        default="config.json",
        help="Path to the configuration file. Default is 'config.json'.")

    args = parser.parse_args()

    # Use the provided config file or fallback to default
    main(args.config)
