import math

import torch


def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.0


def kernel_functional_rbf(losses, scale=0.5):
    n = losses.shape[0]
    pairwise_distance = torch.norm(losses[:, None] - losses.detach(), dim=2)
    h = median(pairwise_distance) / math.log(n)
    h = h.detach()
    kernel_matrix = torch.exp(-pairwise_distance / (scale * h + 1e-8))

    return kernel_matrix


def get_gradient(grads, losses, inputs, mode="svgd"):
    n = grads.size(0)

    with torch.no_grad():
        g_w = MinNormSolver.find_min_norm_element(grads)

    if mode == "linear":
        return g_w / n

    kernel = kernel_functional_rbf(losses)
    kernel_grad_all = None
    for param in inputs:
        kernel_grad = -0.5 * torch.autograd.grad(kernel.sum(), param, allow_unused=True, retain_graph=True)[0]
        if kernel_grad_all is None:
            kernel_grad_all = kernel_grad.flatten()
        else:
            kernel_grad_all = torch.cat([kernel_grad_all, kernel_grad.flatten()])

    kernel_grad_all = kernel_grad_all.view(g_w.shape)

    gradient = kernel.mm(g_w) - kernel_grad_all

    return gradient / n


class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = torch.sum(vecs[i] * vecs[j]).data.cpu()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = torch.sum(vecs[i] * vecs[i]).data.cpu()
                if (j, j) not in dps:
                    dps[(j, j)] = torch.sum(vecs[j] * vecs[j]).data.cpu()
                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    def _projection2simplex(y):
        m = len(y)
        sorted_y = torch.flip(torch.sort(y)[0], dims=[0])
        tmpsum = 0.0
        tmax_f = (y.sum() - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return torch.max(y - tmax_f, torch.zeros(y.shape).cuda())

    def _next_point(cur_val, grad, n):
        proj_grad = grad - (torch.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])
        t = 1

        if len(tm1[tm1 > 1e-7]) > 0:
            t = (tm1[tm1 > 1e-7]).min()
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, (tm2[tm2 > 1e-7]).min())

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):

        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = torch.zeros(n).cuda()
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:

            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = torch.zeros((n, n)).cuda()
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:

            grad_dir = -1.0 * torch.mm(grad_mat, sol_vec.view(-1, 1)).view(-1)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)

            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            # print("Change: ", change)
            try:
                if change.pow(2).sum() < MinNormSolver.STOP_CRIT:
                    return sol_vec, nd
            except Exception as e:
                print(e)
                print("Change: ", change)
                # return sol_vec, nd
            sol_vec = new_sol_vec
        return sol_vec, nd