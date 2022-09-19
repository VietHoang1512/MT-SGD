import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from data import get_data
from models import FCNet
from svgd import MinNormSolver
from tqdm import trange
from utils import count_parameters, get_device, set_seed


@torch.no_grad()
def evaluate(net, loader, device):
    num_nets = len(net)
    for j in range(num_nets):
        net[j].eval()

    full_losses = []
    for batch in loader:

        batch = (t.to(device) for t in batch)
        xs, ys = batch
        preds = []
        for j in range(num_nets):
            preds.append(net[j](xs))
        pred = torch.stack(preds).mean(0)
        curr_losses = get_losses(pred, ys)

        # losses
        full_losses.append(curr_losses.detach().cpu().numpy())
    loss = np.array(full_losses).mean(0)
    return loss.tolist(), loss.mean()


def get_losses(pred, label):
    return F.mse_loss(pred, label, reduction="none").mean(0)


def train(
    path,
    num_nets: int,
    scale: float,
    normalize: bool,
    trade_off: float,
    epochs: int,
    warmup_epochs: int,
    lr: float,
    bs: int,
    device,
) -> None:

    net = net = [FCNet().to(device) for _ in range(num_nets)]

    print("Number of parameters:", count_parameters(net[0]))

    optimizers = [torch.optim.Adam(net[i].parameters(), lr=lr, weight_decay=0) for i in range(num_nets)]
    train_set, val_set, test_set = get_data(path)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)

    epoch_iter = trange(epochs)

    best_val = np.inf
    best_test = np.inf

    for epoch in epoch_iter:

        for i, batch in enumerate(train_loader):

            batch = (t.to(device) for t in batch)
            xs, ys = batch
            num_objectives = ys.shape[1]
            assert num_objectives == 7, "Wrong number of objectives"
            grads = [[] for _ in range(num_nets)]

            losses = []
            outputs = []
            for j in range(num_nets):
                net[j].train()
                net[j].zero_grad()
                out = net[j](xs)
                outputs.append(out.flatten())
                loss = get_losses(out, ys)
                losses.append(loss)
                for k in range(num_objectives):
                    loss[k].backward(retain_graph=True)
                    grad = get_net_gradient(net[j])
                    if normalize:
                        grad = F.normalize(grad, p=2, dim=0)
                    grads[j].append(grad)
                    net[j].zero_grad()
            outputs = torch.stack(outputs)

            losses = torch.stack(losses)

            epoch_iter.set_description(f"MSE: {losses.mean().item():.3f}")

            grads = torch.stack([torch.stack(grad) for grad in grads])

            update_grads = []
            for j in range(num_nets):
                with torch.no_grad():
                    sol, _ = MinNormSolver.find_min_norm_element(grads[j])
                    update_grads.append(sol.unsqueeze(0).mm(grads[j]))
            update_grads = torch.stack(update_grads).squeeze(1)

            if epoch >= warmup_epochs:
                kernel_grads = []

                kernel = RBF(outputs, outputs.detach(), bandwidth_scale=args.scale)
                kernel.sum().backward()
                for j in range(num_nets):
                    kernel_grads.append(get_net_gradient(net[j]))
                    net[j].zero_grad()
                kernel_grads = torch.stack(kernel_grads).squeeze(1)
                kernel_grads = kernel_grads.view(update_grads.shape)

                with torch.no_grad():
                    Q = torch.zeros(num_objectives, num_objectives).cuda()
                    for i in range(num_objectives):
                        for j in range(num_objectives):

                            Q[i][j] = torch.mul(
                                kernel,
                                torch.matmul(grads[:, i, :], grads[:, j, :].T),
                            ).sum()
                            Q[j][i] = Q[i][j]
                    sol, _ = MinNormSolver.find_min_norm_element(Q)
                    sol = sol.unsqueeze(0).repeat(num_nets, 1).unsqueeze(1)

                update_grads = torch.bmm(sol, grads).squeeze(1) + trade_off * kernel_grads

            for j in range(num_nets):
                index = 0
                net[j].zero_grad()
                for name, param in net[j].named_parameters():
                    length = param.grad.flatten().shape[0]

                    cur_grad = update_grads[j, index : index + length].view(param.grad.shape)
                    param.grad.data = cur_grad.data.clone()
                    index += length
                optimizers[j].step()
                net[j].zero_grad()

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


def get_net_gradient(net):
    grads = []
    for _, param in net.named_parameters():
        grads.append(param.grad.detach().clone().flatten())
    return torch.cat(grads)


def get_net_parameters(net):
    params = []
    for _, param in net.named_parameters():
        params.append(param)
    return params


def RBF(X, Y, bandwidth_scale, sigma=None):

    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())

    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

    # Apply the median heuristic (PyTorch does not give true median)
    if sigma is None:
        np_dnorm2 = dnorm2.detach().cpu().numpy()
        h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
        sigma = np.sqrt(h).item() * bandwidth_scale

    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    K_XY = (-gamma * dnorm2).exp()
    return K_XY


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARCOS")

    parser.add_argument("--datapath", type=str, default="data", help="path to data")

    parser.add_argument("--n-epochs", type=int, default=1000, help="num. epochs")

    parser.add_argument("--batch-size", type=int, default=512, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument(
        "--tradeoff",
        default=0,
        type=float,
        help="Stein kernel trade off.",
    )

    parser.add_argument(
        "--scale",
        default=1e-5,
        type=float,
        help="RBF bandwidth scaling factor.",
    )

    parser.add_argument(
        "--num_nets",
        default=5,
        type=int,
        help="Number of particles.",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=100,
        type=int,
        help="Total number of warm up epochs to perform.",
    )

    parser.add_argument("--normalize", action="store_true", help="Whether to normalize.")

    args = parser.parse_args()

    args.output_dir = "outputs/" + str(args).replace(", ", "/").replace("'", "").replace("(", "").replace(
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
        num_nets=args.num_nets,
        scale=args.scale,
        normalize=args.normalize,
        trade_off=args.tradeoff,
        epochs=args.n_epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        bs=args.batch_size,
        device=get_device(),
    )
