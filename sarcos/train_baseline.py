import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from data import get_data
from models import FCNet
from torch import nn
from tqdm import trange
from utils import count_parameters, get_device, set_seed


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()

    full_losses = []
    for batch in loader:

        batch = (t.to(device) for t in batch)
        xs, ys = batch

        pred = net(xs)

        curr_losses = get_losses(pred, ys)

        # losses
        full_losses.append(curr_losses.detach().cpu().numpy())
    loss = np.array(full_losses).mean(0)
    return loss.tolist(), loss.mean()


def get_losses(pred, label):
    return F.mse_loss(pred, label, reduction="none").mean(0)


def train(
    path,
    epochs: int,
    lr: float,
    bs: int,
    device,
) -> None:

    net: nn.Module = FCNet()

    net = net.to(device)
    print("Number of parameters:", count_parameters(net))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)

    train_set, val_set, test_set = get_data(path)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=bs, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=4)

    epoch_iter = trange(epochs)

    best_val = np.inf
    best_test = np.inf

    for epoch in epoch_iter:

        for i, batch in enumerate(train_loader):
            net.train()
            optimizer.zero_grad()
            batch = (t.to(device) for t in batch)
            xs, ys = batch

            pred = net(xs)

            losses = get_losses(pred, ys)

            loss = losses.mean()

            loss.backward()
            epoch_iter.set_description(f"MSE: {losses.mean().item():.3f}")

            optimizer.step()

        val_results, val_results_mean = evaluate(
            net=net,
            loader=val_loader,
            device=device,
        )

        test_results, test_results_mean = evaluate(
            net=net,
            loader=test_loader,
            device=device,
        )
        test_is_best = ""
        val_is_best = ""
        if best_val >= val_results_mean:
            best_val = val_results_mean
            val_is_best = "best validation score!"
        if best_test >= test_results_mean:
            best_test = test_results_mean
            test_is_best = "best test score!"
        logging.info(f"Epoch [{epoch}/{epochs}] - Validation: {val_results_mean:.3f} Test: {test_results_mean:.3f}")
        logging.info(f"Validation: {val_results} {val_is_best}")
        logging.info(f"Test: {test_results} {test_is_best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARCOS")

    parser.add_argument("--datapath", type=str, default="data", help="path to data")
    parser.add_argument("--n-epochs", type=int, default=1000, help="num. epochs")

    parser.add_argument("--batch-size", type=int, default=512, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    args.output_dir = "outputs/baseline/" + str(args).replace(", ", "/").replace("'", "").replace("(", "").replace(
        ")", ""
    ).replace("Namespace", "")

    print("Output directory:", args.output_dir)

    os.system("rm -rf " + args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

    logging.basicConfig(
        filename=f"./{args.output_dir}/sarcos.log",
        level=logging.DEBUG,
        filemode="w",
        datefmt="%H:%M:%S",
        format="%(asctime)s :: %(levelname)-8s \n%(message)s",
    )

    set_seed(args.seed)

    train(
        path=args.datapath,
        epochs=args.n_epochs,
        lr=args.lr,
        bs=args.batch_size,
        device=get_device(),
    )
