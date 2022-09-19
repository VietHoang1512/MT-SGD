import json
import logging
import os
from timeit import default_timer as timer

import click
import datasets
import losses
import metrics
import model_selector
import numpy as np
import torch
from min_norm_solvers import MinNormSolver, gradient_normalizers
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm

NUM_EPOCHS = 100


@click.command()
@click.option("--param_file", default="params.json", help="JSON parameters file")
def train_multi_task(param_file):
    with open("configs.json") as config_params:
        configs = json.load(config_params)

    with open(param_file) as json_params:
        params = json.load(json_params)

    os.makedirs("saved_models/{}".format(params["exp"]), exist_ok=True)

    logging.basicConfig(
        filename="saved_models/{}/train.log".format(params["exp"]),
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

    exp_identifier = "|".join(exp_identifier)
    params["exp_id"] = exp_identifier

    # writer = SummaryWriter(log_dir='runs/{}_{}'.format(params['exp_id'], datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
    writer = SummaryWriter(log_dir="saved_models/{}".format(params["exp"]))
    train_loader, train_dst, val_loader, val_dst = datasets.get_dataset(params, configs)
    loss_fn = losses.get_loss(params)
    metric = metrics.get_metrics(params)

    model = model_selector.get_model(params)
    model_params = []
    for m in model:
        model_params += model[m].parameters()

    if "RMSprop" in params["optimizer"]:
        optimizer = torch.optim.RMSprop(model_params, lr=params["lr"])
    elif "Adam" in params["optimizer"]:
        optimizer = torch.optim.Adam(model_params, lr=params["lr"])
    elif "SGD" in params["optimizer"]:
        optimizer = torch.optim.SGD(model_params, lr=params["lr"], momentum=0.9)

    tasks = params["tasks"]
    all_tasks = configs[params["dataset"]]["all_tasks"]
    logging.info("Starting training with parameters \n \t{} \n".format(str(params)))

    if "mgda" in params["algorithm"]:
        approximate_norm_solution = params["use_approximation"]
        if approximate_norm_solution:
            logging.info("Using approximate min-norm solver")
        else:
            logging.info("Using full solver")
    n_iter = 0
    best_acc = 0
    for epoch in tqdm(range(NUM_EPOCHS)):
        start = timer()
        logging.info("Epoch {} Started".format(epoch))
        if (epoch + 1) % 10 == 0:
            # Every 50 epoch, half the LR
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.85
            logging.info("Half the learning rate{}".format(n_iter))

        for m in model:
            model[m].train()

        for batch in tqdm(train_loader, total=len(train_loader)):
            n_iter += 1
            # First member is always images
            images = batch[0].cuda()

            labels = {}
            # Read all targets of all tasks
            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels[t] = batch[i + 1]
                labels[t] = Variable(labels[t].cuda())

            # Scaling the loss functions based on the algorithm choice
            loss_data = {}
            grads = {}
            scale = {}

            if "mgda" in params["algorithm"]:
                # Will use our MGDA_UB if approximate_norm_solution is True. Otherwise, will use MGDA

                if approximate_norm_solution:
                    optimizer.zero_grad()
                    # First compute representations (z)
                    images_volatile = Variable(images.data)
                    with torch.no_grad():
                        rep = model["rep"](images_volatile)

                    rep_variable = Variable(rep.data.clone(), requires_grad=True)

                    # Compute gradients of each loss function wrt z
                    for t in tasks:
                        optimizer.zero_grad()
                        out_t = model[t](rep_variable)
                        loss = loss_fn[t](out_t, labels[t])
                        loss_data[t] = loss.item()
                        loss.backward()
                        grads[t] = []

                        grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
                        rep_variable.grad.data.zero_()
                else:
                    # This is MGDA
                    for t in tasks:
                        # Comptue gradients of each loss function wrt parameters
                        optimizer.zero_grad()
                        rep = model["rep"](images)
                        out_t = model[t](rep, None)
                        loss = loss_fn[t](out_t, labels[t])
                        loss_data[t] = loss.item()
                        loss.backward()
                        grads[t] = []
                        for param in model["rep"].parameters():
                            if param.grad is not None:
                                grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))

                # Normalize all gradients, this is optional and not included in the paper.
                gn = gradient_normalizers(grads, loss_data, params["normalization_type"])
                for t in tasks:
                    for gr_i in range(len(grads[t])):
                        grads[t][gr_i] = grads[t][gr_i] / gn[t]

                # Frank-Wolfe iteration to compute scales.
                with torch.no_grad():
                    sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
                for i, t in enumerate(tasks):
                    scale[t] = float(sol[i])
            else:
                for t in tasks:
                    scale[t] = float(params["scales"][t])

            # Scaled back-propagation
            optimizer.zero_grad()
            rep = model["rep"](images)
            for i, t in enumerate(tasks):
                out_t = model[t](rep)
                loss_t = loss_fn[t](out_t, labels[t])
                loss_data[t] = loss_t.item()
                if i > 0:
                    loss = loss + scale[t] * loss_t
                else:
                    loss = scale[t] * loss_t
            loss.backward()
            optimizer.step()

            writer.add_scalar("training_loss", loss.item(), n_iter)
            for t in tasks:
                writer.add_scalar("training_loss_{}".format(t), loss_data[t], n_iter)

        for m in model:
            model[m].eval()

        tot_loss = {}
        tot_loss["all"] = 0.0
        met = {}
        for t in tasks:
            tot_loss[t] = 0.0
            met[t] = 0.0

        num_val_batches = 0
        for batch_val in val_loader:
            val_images = Variable(batch_val[0].cuda())
            labels_val = {}

            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels_val[t] = batch_val[i + 1]
                labels_val[t] = Variable(labels_val[t].cuda())
            with torch.no_grad():
                val_rep = model["rep"](val_images)
                for t in tasks:
                    out_t_val = model[t](val_rep)
                    loss_t = loss_fn[t](out_t_val, labels_val[t])
                    tot_loss["all"] += loss_t.item()
                    tot_loss[t] += loss_t.item()
                    metric[t].update(out_t_val, labels_val[t])
                num_val_batches += 1
        accs = []
        for t in tasks:
            writer.add_scalar("validation_loss_{}".format(t), tot_loss[t] / num_val_batches, n_iter)
            metric_results = metric[t].get_result()
            for metric_key in metric_results:
                writer.add_scalar("metric_{}_{}".format(metric_key, t), metric_results[metric_key], n_iter)
                accs.append(metric_results[metric_key])
            metric[t].reset()
        acc = np.mean(accs)
        writer.add_scalar("validation_loss", tot_loss["all"] / len(val_dst), n_iter)

        if acc > best_acc:
            logging.info("Accuracy improved from {} to {}".format(best_acc, acc))
            best_acc = acc
            state = {"epoch": epoch + 1, "model_rep": model["rep"].state_dict()}
            for t in tasks:
                key_name = "model_{}".format(t)
                state[key_name] = model[t].state_dict()

            torch.save(state, f"saved_models/{params['exp']}/best_model.pt")

        end = timer()
        logging.info("Epoch ended in {}s".format(end - start))


if __name__ == "__main__":
    train_multi_task()
