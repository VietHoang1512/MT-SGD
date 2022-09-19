import argparse
import logging
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from model_lenet import RegressionModel
from reliability_diagrams import reliability_diagram
from svgd import MinNormSolver, get_gradient

logging.getLogger("matplotlib").setLevel(logging.WARNING)

plt.style.use("seaborn")

plt.rc("font", size=12)
plt.rc("axes", labelsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("legend", fontsize=12)

plt.rc("axes", titlesize=16)
plt.rc("figure", titlesize=16)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dset",
    default="multi_fashion_and_mnist",
    required=True,
    type=str,
    help="Dataset for training.",
)

parser.add_argument(
    "--output_scale",
    default=1e-5,
    type=float,
    help="RBF bandwidth scaling factor on outputs.",
)

parser.add_argument(
    "--latent_scale",
    default=1e-5,
    type=float,
    help="RBF bandwidth scaling factor on latents.",
)

parser.add_argument(
    "--num_nets",
    default=5,
    type=int,
    help="Number of particles.",
)

parser.add_argument(
    "--batch_size",
    default=256,
    type=int,
    help="Batch size.",
)

parser.add_argument("--lr", default=1e-3, type=float, help="The initial learning rate for SGD .")

parser.add_argument(
    "--output_tradeoff",
    default=10,
    type=float,
    help="Stein kernel trade offon outputs.",
)

parser.add_argument(
    "--latent_tradeoff",
    default=10,
    type=float,
    help="Stein kernel trade off on latents.",
)


parser.add_argument(
    "--n_epochs",
    default=100,
    type=int,
    help="Total number of training epochs to perform.",
)

parser.add_argument(
    "--warmup_epoch",
    default=1,
    type=int,
    help="Total number of warm up epochs to perform.",
)

parser.add_argument("--normalize", action="store_true", help="Whether to normalize.")

parser.add_argument("--seed", type=int, default=0, help="seed")

args = parser.parse_args()

args.output_dir = "outputs/" + str(args).replace(", ", "/").replace("'", "").replace("(", "").replace(")", "").replace(
    "Namespace", ""
)

print("Output directory:", args.output_dir)
os.system("rm -rf " + args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "config.yaml"), "w") as outfile:
    yaml.dump(vars(args), outfile, default_flow_style=False)

logging.basicConfig(
    filename=f"./{args.output_dir}/{args.dset}.log",
    level=logging.DEBUG,
    filemode="w",
    datefmt="%H:%M:%S",
    format="%(asctime)s :: %(levelname)-8s \n%(message)s",
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


setup_seed(args.seed)

with open(f"./data/{args.dset}.pickle", "rb") as f:
    trainX, trainLabel, testX, testLabel = pickle.load(f)
trainX = torch.from_numpy(trainX.reshape(120000, 1, 36, 36)).float()
trainLabel = torch.from_numpy(trainLabel).long()
testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
testLabel = torch.from_numpy(testLabel).long()
train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
test_set = torch.utils.data.TensorDataset(testX, testLabel)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
logging.info("==>>> total trainning batch number: {}".format(len(train_loader)))
logging.info("==>>> total testing batch number: {}".format(len(test_loader)))


# Random-batch-training
criterion = nn.CrossEntropyLoss()

net = [RegressionModel(2).cuda() for _ in range(args.num_nets)]

param_amount = 0
for p in net[0].named_parameters():
    param_amount += p[1].numel()
    print(p[0], p[1].numel())
logging.info(f"total param amount: {param_amount}")

shared_optimizers = [
    torch.optim.SGD(net[i].get_shared_parameters(), lr=args.lr, momentum=0.9) for i in range(args.num_nets)
]

classifier_optimizers = [
    torch.optim.SGD(net[i].get_classifier_parameters(), lr=args.lr, momentum=0.9) for i in range(args.num_nets)
]

shared_schedulers = [
    torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75, 90], gamma=0.5)
    for optimizer in shared_optimizers
]
classifier_schedulers = [
    torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75, 90], gamma=0.5)
    for optimizer in classifier_optimizers
]


def train(epoch):

    if epoch > 0:
        for i in range(args.num_nets):
            shared_schedulers[i].step()
            classifier_optimizers[i].step()

    # training
    all_losses_1 = [0.0 for i in range(args.num_nets)]
    all_losses_2 = [0.0 for i in range(args.num_nets)]
    for (it, batch) in enumerate(train_loader):
        X = batch[0]
        y = batch[1]
        X, y = X.cuda(), y.cuda()
        batchsize_cur = X.shape[0]

        score_grads1_all = []
        score_grads2_all = []
        features = []
        outputs1 = []
        outputs2 = []

        for i in range(args.num_nets):
            net[i].train()
            net[i].zero_grad()
            features.append(net[i].encode(X))
            ################################## Classifier ##################################
            out1, out2 = net[i].decode(features[i].detach().clone())

            loss1 = criterion(out1, y[:, 0])
            all_losses_1[i] += loss1.detach().cpu().numpy() * batchsize_cur
            loss1.backward(retain_graph=True)
            loss2 = criterion(out2, y[:, 1])
            all_losses_2[i] += loss2.detach().cpu().numpy() * batchsize_cur
            loss2.backward(retain_graph=True)

            score_grads1 = None
            score_grads2 = None
            for name, param in net[i].named_parameters():
                if "task_1" in name:
                    if score_grads1 is None:
                        score_grads1 = param.grad.detach().data.clone().flatten()
                    else:
                        score_grads1 = torch.cat([score_grads1, param.grad.detach().data.clone().flatten()])

                if "task_2" in name:
                    if score_grads2 is None:
                        score_grads2 = param.grad.detach().data.clone().flatten()
                    else:
                        score_grads2 = torch.cat([score_grads2, param.grad.detach().data.clone().flatten()])
            net[i].zero_grad()
            outputs1.append(out1.reshape(-1))
            outputs2.append(out2.reshape(-1))

            score_grads1_all.append(score_grads1)
            score_grads2_all.append(score_grads2)

        score_grads1_all = torch.stack(score_grads1_all)
        score_grads2_all = torch.stack(score_grads2_all)

        # score_grads1_all = torch.nn.functional.normalize(score_grads1_all, dim=0)
        # score_grads2_all = torch.nn.functional.normalize(score_grads2_all, dim=0)

        outputs1 = torch.stack(outputs1, dim=0)
        outputs2 = torch.stack(outputs2, dim=0)

        kernel1 = RBF(outputs1, outputs1.detach(), args.output_scale)
        kernel1.sum().backward()
        kernel2 = RBF(outputs2, outputs2.detach(), args.output_scale)
        kernel2.sum().backward()
        # print("kernel1", kernel1)
        # print("kernel2", kernel2)
        kernel_grads1_all = []
        kernel_grads2_all = []
        for i in range(args.num_nets):
            kernel_grads1 = None
            kernel_grads2 = None

            for name, param in net[i].named_parameters():
                if "task_1" in name:
                    if kernel_grads1 is None:
                        kernel_grads1 = param.grad.detach().data.clone().flatten()
                    else:
                        kernel_grads1 = torch.cat([kernel_grads1, param.grad.detach().data.clone().flatten()])

                if "task_2" in name:
                    if kernel_grads2 is None:
                        kernel_grads2 = param.grad.detach().data.clone().flatten()
                    else:
                        kernel_grads2 = torch.cat([kernel_grads2, param.grad.detach().data.clone().flatten()])
            net[i].zero_grad()
            kernel_grads1_all.append(kernel_grads1)
            kernel_grads2_all.append(kernel_grads2)

        kernel_grads1_all = torch.stack(kernel_grads1_all)
        kernel_grads2_all = torch.stack(kernel_grads2_all)

        # print("kernel_grads1_all", kernel_grads1_all)
        # print("kernel_grads2_all", kernel_grads2_all)

        gradient1 = (kernel1.mm(score_grads1_all) + args.output_tradeoff * kernel_grads1_all) / args.num_nets
        gradient2 = (kernel2.mm(score_grads2_all) + args.output_tradeoff * kernel_grads2_all) / args.num_nets

        if epoch < args.warmup_epoch:
            gradient1 = score_grads1_all
            gradient2 = score_grads2_all
        # gradient1 = score_grads1_all
        # gradient2 = score_grads2_all
        # print("Task 1 gradient", gradient1.mean(0))
        # print("Task 2 gradient", gradient2.mean(0))
        score_grads1_all = []
        score_grads2_all = []
        for i in range(args.num_nets):
            index1 = 0
            index2 = 0
            net[i].zero_grad()
            for name, param in net[i].named_parameters():
                if "task_1" in name:

                    length = param.grad.flatten().shape[0]
                    cur_grad = gradient1[i, index1 : index1 + length].view(param.grad.shape)
                    param.grad.data = cur_grad.data.clone()
                    index1 += length

                if "task_2" in name:

                    length = param.grad.flatten().shape[0]
                    cur_grad = gradient2[i, index2 : index2 + length].view(param.grad.shape)
                    param.grad.data = cur_grad.data.clone()
                    index2 += length
            assert index1 == gradient1.shape[1] and index2 == gradient1.shape[1], "Redundant gradient"
            classifier_optimizers[i].step()
            net[i].zero_grad()
            ############################## Encoder ##################################
            out1, out2 = net[i].decode(features[i])
            loss1 = criterion(out1, y[:, 0])
            loss1.backward(retain_graph=True)

            grads1 = None
            for name, param in net[i].named_parameters():
                if "task" not in name:
                    if grads1 is None:
                        grads1 = param.grad.detach().data.clone().flatten()
                    else:
                        grads1 = torch.cat([grads1, param.grad.detach().data.clone().flatten()])
                param.grad.zero_()
            loss2 = criterion(out2, y[:, 1])
            loss2.backward(retain_graph=True)
            grads2 = None
            for name, param in net[i].named_parameters():
                if "task" not in name:
                    if grads2 is None:
                        grads2 = param.grad.detach().data.clone().flatten()
                    else:
                        grads2 = torch.cat([grads2, param.grad.detach().data.clone().flatten()])
                param.grad.zero_()
            score_grads1_all.append(grads1)
            score_grads2_all.append(grads2)
            net[i].zero_grad()

        features = torch.stack([feature.flatten() for feature in features], dim=0)

        kernel = RBF(features, features.detach(), args.latent_scale)
        kernel.sum().backward()

        kernel_grads_all = []
        for i in range(args.num_nets):
            kernel_grads = None
            for name, param in net[i].named_parameters():
                if "task" not in name:
                    if kernel_grads is None:
                        kernel_grads = param.grad.detach().data.clone().flatten()
                    else:
                        kernel_grads = torch.cat([kernel_grads, param.grad.detach().data.clone().flatten()])
                param.grad.zero_()
            kernel_grads_all.append(kernel_grads)
            net[i].zero_grad()

        kernel_grads_all = torch.stack(kernel_grads_all)

        # process the SVGD gradient
        score_grads1_all = torch.cat([score_grads1_all[i].unsqueeze(0) for i in range(args.num_nets)], dim=0)
        score_grads2_all = torch.cat([score_grads2_all[i].unsqueeze(0) for i in range(args.num_nets)], dim=0)
        if args.normalize:
            score_grads1_all = torch.nn.functional.normalize(score_grads1_all, dim=0)
            score_grads2_all = torch.nn.functional.normalize(score_grads2_all, dim=0)

        gradient1 = (kernel.mm(score_grads1_all) + args.latent_tradeoff * kernel_grads_all) / args.num_nets
        gradient2 = (kernel.mm(score_grads2_all) + args.latent_tradeoff * kernel_grads_all) / args.num_nets

        Q = torch.zeros((2, 2))
        # print("kernel", kernel)
        # print("kernel_grads_all", kernel_grads_all)
        # print("torch.matmul(score_grads1_all, score_grads2_all.T)", torch.matmul(score_grads1_all, score_grads2_all.T).shape)
        with torch.no_grad():
            Q[0][1] = torch.mul(kernel, torch.matmul(score_grads1_all, score_grads2_all.T)).sum()
            Q[1][0] = Q[0][1]
            Q[0][0] = torch.mul(kernel, torch.matmul(score_grads1_all, score_grads1_all.T)).sum()
            Q[1][1] = torch.mul(kernel, torch.matmul(score_grads2_all, score_grads2_all.T)).sum()
            # Q = Q/Q.sum()
            sol, _ = MinNormSolver.find_min_norm_element(Q)
            # print("Q", Q)
            # print("sol", sol)
            gradient = gradient1 * sol[0].item() + gradient2 * sol[1].item()
            # gradient = (gradient1 + gradient2)/2.0

        if epoch < args.warmup_epoch:
            # weight = np.random.uniform()
            # gradient = weight * score_grads1_all + (1 - weight) * score_grads2_all
            gradient, _ = get_gradient(score_grads1_all, score_grads2_all, None, None, None, "linear")

        # print("Encoder grad", gradient)
        for i in range(args.num_nets):
            index = 0
            net[i].zero_grad()
            for name, param in net[i].named_parameters():
                if "task" not in name:
                    length = param.grad.flatten().shape[0]
                    cur_grad = gradient[i, index : index + length].view(param.grad.shape)
                    param.grad.data = cur_grad.data.clone()
                    index += length
            assert index == gradient.shape[1], "Redundant gradient"
            shared_optimizers[i].step()
            net[i].zero_grad()

    all_losses_1 = np.array(all_losses_1)[:, np.newaxis]
    all_losses_2 = np.array(all_losses_2)[:, np.newaxis]

    losses = np.concatenate((all_losses_1, all_losses_2), axis=1) / len(train_loader.dataset)
    logging.info(f"TRAIN TRAIN LOSS:\n{losses}")

    return losses


@torch.no_grad()
def eval_train(epoch):
    # training
    all_losses_1 = [0.0 for i in range(args.num_nets)]
    all_losses_2 = [0.0 for i in range(args.num_nets)]
    for (it, batch) in enumerate(train_loader):
        X = batch[0]
        y = batch[1]
        X, y = X.cuda(), y.cuda()
        batchsize_cur = X.shape[0]

        for i in range(args.num_nets):
            net[i].eval()
            out1, out2 = net[i](X)

            loss1 = criterion(out1, y[:, 0])
            all_losses_1[i] += loss1.detach().cpu().numpy() * batchsize_cur

            loss2 = criterion(out2, y[:, 1])
            all_losses_2[i] += loss2.detach().cpu().numpy() * batchsize_cur

    all_losses_1 = np.array(all_losses_1)[:, np.newaxis]
    all_losses_2 = np.array(all_losses_2)[:, np.newaxis]

    losses = np.concatenate((all_losses_1, all_losses_2), axis=1) / len(train_loader.dataset)
    logging.info(f"EVAL TRAIN LOSS: \n{losses}")

    return losses


@torch.no_grad()
def test():
    all_acc_1 = torch.zeros(args.num_nets).cuda()
    all_acc_2 = torch.zeros(args.num_nets).cuda()
    for i in range(args.num_nets):
        net[i].eval()

    acc_1_ensemble = 0
    acc_2_ensemble = 0
    task1_ground_truth = []
    task2_ground_truth = []
    task1_preds = []
    task2_preds = []
    with torch.no_grad():

        for (it, batch) in enumerate(test_loader):
            X = batch[0]
            y = batch[1]
            X = X.cuda()
            y = y.cuda()
            out1_probs = []
            out2_probs = []
            for i in range(args.num_nets):
                out1_prob, out2_prob = net[i](X)
                out1_prob = F.softmax(out1_prob, dim=1)
                out2_prob = F.softmax(out2_prob, dim=1)
                out1_probs.append(out1_prob)
                out2_probs.append(out2_prob)
                out1 = out1_prob.max(1)[1]
                out2 = out2_prob.max(1)[1]
                all_acc_1[i] += (out1 == y[:, 0]).sum()
                all_acc_2[i] += (out2 == y[:, 1]).sum()

            out1_prob = torch.stack(out1_probs).mean(0)
            out2_prob = torch.stack(out2_probs).mean(0)
            task1_ground_truth.append(y[:, 0].detach().clone())
            task2_ground_truth.append(y[:, 1].detach().clone())
            task1_preds.append(out1_prob.detach().clone())
            task2_preds.append(out2_prob.detach().clone())
            out1 = out1_prob.max(1)[1]
            out2 = out2_prob.max(1)[1]
            acc_1_ensemble += (out1 == y[:, 0]).sum()
            acc_2_ensemble += (out2 == y[:, 1]).sum()

        all_acc_1 = all_acc_1.cpu().numpy()[:, np.newaxis]
        all_acc_2 = all_acc_2.cpu().numpy()[:, np.newaxis]

        acc = np.concatenate((all_acc_1, all_acc_2), axis=1) / len(test_loader.dataset)
        acc_1_ensemble = acc_1_ensemble.item() / len(test_loader.dataset)
        acc_2_ensemble = acc_2_ensemble.item() / len(test_loader.dataset)
        task1_preds = torch.cat(task1_preds)
        task2_preds = torch.cat(task2_preds)
        task1_confidence, task1_preds = torch.max(task1_preds, 1)
        task2_confidence, task2_preds = torch.max(task2_preds, 1)

        task1_ground_truth = torch.cat(task1_ground_truth).cpu().numpy()
        task2_ground_truth = torch.cat(task2_ground_truth).cpu().numpy()

        fig1 = reliability_diagram(
            task1_ground_truth,
            task1_preds.cpu().numpy(),
            task1_confidence.cpu().numpy(),
            num_bins=10,
            draw_ece=True,
            draw_bin_importance="alpha",
            draw_averages=True,
            title="Task 1 Expected Calibration Error",
            figsize=(6, 6),
            dpi=100,
            return_fig=True,
        )
        fig2 = reliability_diagram(
            task2_ground_truth,
            task2_preds.cpu().numpy(),
            task2_confidence.cpu().numpy(),
            num_bins=10,
            draw_ece=True,
            draw_bin_importance="alpha",
            draw_averages=True,
            title="Task 2 Expected Calibration Error",
            figsize=(6, 6),
            dpi=100,
            return_fig=True,
        )

        logging.info(f"TEST ACCURACY: \n{acc}")
        logging.info(f"ENSEMBLE TEST ACCURACY:\n{acc_1_ensemble}, {acc_2_ensemble}")

    return (acc_1_ensemble + acc_2_ensemble) / 2.0, fig1, fig2


best_score = 0
for i in range(args.n_epochs):
    logging.info(f"Epoch [{i}/{args.n_epochs}]")
    losses = train(i)
    losses = eval_train(i)
    score, fig1, fig2 = test()
    if score > best_score:
        logging.info(f"Score improved from {best_score} to {score}")
        fig1.savefig(f"{args.output_dir}/task1_reliability.pdf", bbox_inches="tight")
        fig2.savefig(f"{args.output_dir}/task2_reliability.pdf", bbox_inches="tight")
        for j in range(args.num_nets):
            torch.save(net[j].state_dict(), f"{args.output_dir}/net_svgd_%d.pt" % j)

        best_score = score
    plt.close()
