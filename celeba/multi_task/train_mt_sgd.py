import argparse
import json
import logging
import os
from timeit import default_timer as timer

import datasets
import losses
import metrics
import model_selector
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import RBF, MinNormSolver, get_net_gradient

NUM_EPOCHS = 100


parser = argparse.ArgumentParser()

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

args = parser.parse_args()


def expected_calibration_error(y_true, y_pred, num_bins=15):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]


OUTPUR_DIR = "saved_models_ece/mt-sgd-full/mean_prob/" + str(args).replace(", ", "/").replace("'", "").replace(
    "(", ""
).replace(")", "").replace("Namespace", "")
os.makedirs(OUTPUR_DIR, exist_ok=True)


with open("configs.json") as config_params:
    configs = json.load(config_params)

with open("mt-sgd-full.json") as json_params:
    params = json.load(json_params)

logging.basicConfig(
    filename="{}/train.log".format(OUTPUR_DIR),
    level=logging.DEBUG,
    filemode="w",
    datefmt="%H:%M:%S",
    format="%(asctime)s :: %(levelname)-8s \n%(message)s",
)
exp_identifier = []
for (key, val) in params.items():
    if "tasks" in key:
        continue
    exp_identifier += ["{}={}".format(key, val)]

writer = SummaryWriter(log_dir=OUTPUR_DIR)
train_loader, train_dst, val_loader, val_dst = datasets.get_dataset(params, configs)
loss_fn = losses.get_loss(params)
metric = metrics.get_metrics(params)

num_nets = args.num_nets
model = model_selector.get_model(params, num_nets)

tasks = params["tasks"]
num_tasks = len(tasks)
all_tasks = configs[params["dataset"]]["all_tasks"]
logging.info("Starting training with parameters \n \t{} \n".format(str(params)))

output_scale = args.output_scale
latent_scale = args.latent_scale
output_tradeoff = args.output_tradeoff
latent_tradeoff = args.latent_tradeoff

shared_model_params = []
for m in model["rep"]:
    shared_model_params += m.parameters()

classifier_model_params = []
for t in tasks:
    for m in model[t]:
        classifier_model_params += m.parameters()

shared_optimizers = [torch.optim.Adam(m.parameters(), lr=params["lr"]) for m in model["rep"]]
classifier_optimizers = {t: [torch.optim.Adam(m.parameters(), lr=params["lr"]) for m in model[t]] for t in tasks}

n_iter = 0
best_acc = 0
for epoch in tqdm(range(NUM_EPOCHS)):
    start = timer()
    logging.info("Epoch {} Started".format(epoch))

    if (epoch + 1) % 10 == 0:
        for j in range(num_nets):
            for param_group in shared_optimizers[j].param_groups:
                param_group["lr"] *= 0.85
            for t in tasks:
                for param_group in classifier_optimizers[t][j].param_groups:
                    param_group["lr"] *= 0.85
        logging.info("Half the learning rate{}".format(n_iter))

    for t in model:
        for m in model[t]:
            m.train()
            m.zero_grad()

    for batch in tqdm(train_loader, total=len(train_loader), desc="Training"):
        n_iter += 1
        # First member is always images
        images = batch[0].cuda()

        labels = {}
        # Read all targets of all tasks
        for i, t in enumerate(all_tasks):
            if t not in tasks:
                continue
            labels[t] = batch[i + 1].cuda()

        loss_data = {}
        reps = []
        reps_detach = []
        for m in model["rep"]:
            rep = m(images)
            reps.append(rep)
            reps_detach.append(rep.detach().clone())
        reps = torch.stack(reps, dim=0)

        ########## SVGD for classifiers ##########
        for i, t in enumerate(tasks):
            classifier_score_grads = []
            classifier_outs = []
            loss_data[t] = 0
            for j, m in enumerate(model[t]):
                out = m(reps_detach[j])
                classifier_outs.append(out.reshape(-1))
                loss = loss_fn[t](out, labels[t])
                loss_data[t] += loss.item()
                loss.backward(retain_graph=True)
                classifier_score_grads.append(get_net_gradient(m))
                m.zero_grad()
            classifier_score_grads = torch.stack(classifier_score_grads)
            classifier_outs = torch.stack(classifier_outs)
            classifier_kernel = RBF(classifier_outs, classifier_outs.detach(), bandwidth_scale=output_scale)
            classifier_kernel.sum().backward()
            classifier_kernel_grads = []
            for m in model[t]:
                classifier_kernel_grads.append(get_net_gradient(m))
                m.zero_grad()
            classifier_kernel_grads = torch.stack(classifier_kernel_grads)
            classifier_gradient = (
                classifier_kernel.mm(classifier_score_grads) + output_tradeoff * classifier_kernel_grads
            ) / num_nets
            for j, m in enumerate(model[t]):
                index = 0
                for name, param in m.named_parameters():
                    length = param.grad.flatten().shape[0]
                    cur_grad = classifier_gradient[j, index : index + length].view(param.grad.shape)
                    param.grad.data = cur_grad.data.clone()
                    index += length
                assert index == classifier_score_grads.shape[1], "Redundant gradient"
                classifier_optimizers[t][j].step()
                m.zero_grad()

        ########## SVGD for shared encoder ##########

        for m in model["rep"]:
            m.zero_grad()

        shared_score_grads = []

        for i, t in enumerate(tasks):
            shared_score_grad = []
            for j in range(num_nets):
                out = model[t][j](reps[j])
                loss = loss_fn[t](out, labels[t])
                loss.backward(retain_graph=True)
                grad = get_net_gradient(model["rep"][j])
                grad = F.normalize(grad, p=2, dim=0)
                shared_score_grad.append(grad)
                model["rep"][j].zero_grad()
            shared_score_grad = torch.stack(shared_score_grad)
            shared_score_grads.append(shared_score_grad)
        shared_score_grads = torch.stack(shared_score_grads)

        reps = torch.stack([rep.flatten() for rep in reps], dim=0)
        shared_kernel = RBF(reps, reps.detach(), bandwidth_scale=latent_scale)

        shared_kernel.sum().backward()
        shared_kernel_grads = []

        for m in model["rep"]:
            grad = get_net_gradient(m)
            # grad = F.normalize(grad, p=2, dim=0)
            shared_kernel_grads.append(grad)
            m.zero_grad()
        shared_kernel_grads = torch.stack(shared_kernel_grads)

        with torch.no_grad():
            Q = torch.zeros(num_tasks, num_tasks).cuda()
            for i in range(num_tasks):
                for j in range(i, num_tasks):

                    Q[i][j] = torch.mul(
                        shared_kernel, torch.matmul(shared_score_grads[i], shared_score_grads[j].T)
                    ).sum()
                    Q[j][i] = Q[i][j]
            sol, _ = MinNormSolver.find_min_norm_element(Q)
            sol = sol.unsqueeze(1)
            shared_score_grads = torch.sum(shared_score_grads * sol.unsqueeze(1), dim=0)

        shared_gradient = (shared_kernel.mm(shared_score_grads) + latent_tradeoff * shared_kernel_grads) / num_nets

        for j in range(num_nets):
            index = 0
            for name, param in model["rep"][j].named_parameters():

                length = param.grad.flatten().shape[0]
                cur_grad = shared_gradient[j, index : index + length].view(param.grad.shape)
                param.grad.data = cur_grad.data.clone()
                index += length
            assert index == shared_score_grads.shape[1], "Redundant gradient"
            shared_optimizers[j].step()
            model["rep"][j].zero_grad()

        writer.add_scalar("training_loss", loss.item(), n_iter)
        for t in tasks:
            writer.add_scalar("training_loss_{}".format(t), loss_data[t], n_iter)

    for t in model:
        for j in range(num_nets):
            model[t][j].zero_grad()
            model[t][j].eval()

    tot_loss = {}
    tot_loss["all"] = 0.0
    met = {}
    for t in tasks:
        tot_loss[t] = 0.0
        met[t] = 0.0
    out_pred = {t: [] for t in tasks}
    out_true = {t: [] for t in tasks}
    num_val_batches = 0
    for batch_val in val_loader:
        val_images = batch_val[0].cuda()
        labels_val = {}

        for i, t in enumerate(all_tasks):
            if t not in tasks:
                continue
            labels_val[t] = batch_val[i + 1].cuda()
        with torch.no_grad():
            val_reps = []
            for m in model["rep"]:
                val_reps.append(m(val_images))
            for t in tasks:
                out_vals = []
                for j in range(num_nets):
                    out_vals.append(model[t][j](val_reps[j]))
                out_val = torch.exp(torch.stack(out_vals)).mean(0)
                out_val = torch.log(out_val)
                # out_val = torch.stack(out_vals).mean(0)
                out_pred[t].append(torch.exp(out_val.data).cpu().numpy())
                out_true[t].append(labels_val[t].data.cpu().numpy())
                loss_t = loss_fn[t](out_val, labels_val[t])
                tot_loss["all"] += loss_t.item()
                tot_loss[t] += loss_t.item()
                metric[t].update(out_val, labels_val[t])
            num_val_batches += 1
    accs = []
    eces = []
    for t in tasks:
        out_pred[t] = np.concatenate(out_pred[t])
        out_true[t] = np.concatenate(out_true[t])
        writer.add_scalar("validation_loss_{}".format(t), tot_loss[t] / num_val_batches, n_iter)
        metric_results = metric[t].get_result()
        ece = expected_calibration_error(out_true[t], out_pred[t], num_bins=10)
        eces.append(ece)
        logging.info(f"{t} ECE: {ece}")
        for metric_key in metric_results:
            writer.add_scalar("metric_{}_{}".format(metric_key, t), metric_results[metric_key], n_iter)
            accs.append(metric_results[metric_key])

        metric[t].reset()
    assert len(accs) == len(tasks), "Wrong number of tasks or metrics"
    acc = np.mean(accs)
    log_str = "Validation accuracy {}: {}".format(acc, accs)
    logging.info(log_str)
    print(log_str)
    ece = np.mean(eces)
    log_str = "Validation ECE {}: {}".format(ece, eces)
    logging.info(log_str)
    print(log_str)
    if acc >= best_acc:
        log_str = "Accuracy improved from {} to {}".format(best_acc, acc)
        logging.info(log_str)
        best_acc = acc
        state = {"epoch": epoch + 1, "model_rep": [m.state_dict() for m in model["rep"]]}
        state["out_pred"] = out_pred
        state["out_true"] = out_true
        for t in tasks:
            key_name = "model_{}".format(t)
            state[key_name] = [m.state_dict() for m in model[t]]

        torch.save(state, f"{OUTPUR_DIR}/best_model.pt")
    writer.add_scalar("validation_loss", tot_loss["all"] / len(val_dst), n_iter)

    end = timer()
    logging.info("Epoch ended in {}s".format(end - start))
