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


def train_model(model,
                lr,
                train_dataloader,
                val_dataloader,
                verbose=True,
                epochs=2,
                lambda_reg=0.01,
                pos_weight=1 / 9,
                mu_reg=1,
                criterion="min"):
    model = copy.copy(model)
    best_model = model
    val_acc = []
    best_acc = 0
    train_loss = []
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, batch in tq(enumerate(train_dataloader),
                           verbose,
                           desc=f"Epoch {epoch+1}",
                           total=len(train_dataloader)):
            optim.zero_grad()
            x, y = batch
            y_hat = model(x)
            l1_norm = sum(p.abs().sum() for p in model.parameters())

            loss = loss_fn(y_hat, y.unsqueeze(-1).float())
            loss += lambda_reg * l1_norm
            epoch_loss += loss.item()
            loss.backward()
            optim.step()

        epoch_loss /= len(train_dataloader)
        acc0, acc1, acc_tot = validate_model(model, val_dataloader)
        if criterion == "min":
            acc = min(acc0, acc1)
        if criterion == "tot":
            acc = acc_tot
        else:
            acc = acc_tot
        train_loss.append(epoch_loss)
        val_acc.append(acc)
        if acc > best_acc:
            print("best")
            best_acc = acc
            best_model = copy.copy(model)

        if verbose:
            print(f"""Epoch {epoch+1}, loss : {epoch_loss},
                  acc_an : {acc0:.2f}, acc_na : {acc1:.2f}, acc_tot : {acc_tot:.2f}"""
                  )

    return best_model, model, train_loss, val_acc


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        return config


def main(config_file, ablation):
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

    train_dataloader = DataLoader(train_ds,
                                  batch_size=config['batch_size'],
                                  shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=config['batch_size'])
    test_dataloader = DataLoader(test_ds, batch_size=config['batch_size'])

    model, best_model, train_loss, val_loss = train_model(
        op,
        lr=config['learning_rate'],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=config['num_epochs'],
        verbose=True,
        pos_weight=config['positive_weight'],
        lambda_reg=config['lambda_reg'],
        mu_reg=config['mu_reg'])

    print("Train losses :", [f"{l:.2f} " for l in train_loss])
    print("Val scores :", [f"{l:.2f} " for l in val_loss])
    test_acc = validate_model(best_model, test_dataloader)
    print(
        f"Test Accuracy: an : {test_acc[0]:.2f}, na : {test_acc[1]:.2f}, total : {test_acc[2]:.2f}"
    )

    print("Weights")
    print(best_model.batch_norm.weight)
    print("Biases")
    print(best_model.batch_norm.bias)
    torch.save(best_model.state_dict(), "models/best.pt")
    if ablation:
        for ablation_mask in [(1,0),(0,1)]:
            op = OrderPredictor(embeddingModel=fasttext.load_model(
                config['embeddingModel']),
                num_layers=config['num_layers'],
                hidden_dim=config['hidden_dim'],
                dropout=config['dropout'], ablation_mask=ablation_mask)
            print(f"Ablation mask noun, adjective: {ablation_mask}")
            model, best_model, train_loss, val_loss = train_model(
                op,
                lr=config['learning_rate'],
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                epochs=config['num_epochs'],
                verbose=True,
                lambda_reg=config['lambda_reg'],
                mu_reg=config['mu_reg'])
            print("Train losses :", [f"{l:.2f} " for l in train_loss])
            print("Val scores :", [f"{l:.2f} " for l in val_loss])
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
    parser.add_argument('--ablation',type=bool, default=False)


    args = parser.parse_args()

    # Use the provided config file or fallback to default
    main(args.config, args.ablation )
