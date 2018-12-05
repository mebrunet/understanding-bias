"""Simple script to test the speed of computing HVPs"""

from time import time
import torch
from torch import nn
from torch.autograd import grad

CUDA = torch.cuda.is_available()
print("Using CUDA:", CUDA)

# V = 213_687
# D = 200
# nnz = 525_393_925  # Number of non-zero cooc entries in NYT (min vocab = 15, window = 8)
# fill_factor = nnz / V / V  # 0.011

torch.min(torch.rand(10), torch.ones(1))


# Define the model and loss
class GloVe(nn.Module):
    def __init__(self, V, D):
        super(GloVe, self).__init__()
        self.W = nn.Parameter(torch.randn(V, D))
        self.b_w = nn.Parameter(torch.randn(V))
        self.U = nn.Parameter(torch.randn(V, D))
        self.b_u = nn.Parameter(torch.randn(V))
        self.threshold = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.alpha = nn.Parameter(torch.tensor(0.75), requires_grad=False)
        self.x_max = nn.Parameter(torch.tensor(100.0), requires_grad=False)

    def forward(self, indices):
        I, J = indices
        dot_prods = torch.einsum('nd,nd->n', (self.W[I], self.U[J]))
        biases = self.b_w[I] + self.b_u[J]
        return dot_prods + biases

    def loss(self, data_batch):
        indices, coocs = data_batch
        weights = torch.min((coocs / self.x_max) ** self.alpha, self.threshold)
        outputs = self.forward(indices)
        return torch.mean(weights * (outputs - torch.log(coocs))**2)


# Generate data
def make_batch(N, V):
    indices = torch.randint(V, (2, N), dtype=torch.long)
    coocs = 200 * torch.rand(N)
    if CUDA:
        indices = indices.cuda()
        coocs = coocs.cuda()
    return (indices, coocs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def hvp(loss, params, vectors):
    r"""Computes the hessian vector product.
    Exploits linearity of the gradient. i.e.
        hessian(loss) @ v == grad(grad(loss) @ v)
    Arguments:
        loss (singleton Tensor): the loss evaluated after a forward pass
        params (sequence of Tensor): Inputs w.r.t. which the gradients will be
            returned (and not accumulated into ``.grad``).
        vectors (sequence of Tensor): Vector which the hessian is to be
            multiplied by. Should be of the same length and size(s) as params.
    """

    params = (params,) if isinstance(params, torch.Tensor) else tuple(params)
    vectors = (vectors,) if isinstance(vectors, torch.Tensor) else tuple(vectors)
    grads = grad(loss, params, create_graph=True)
    prod_sums = tuple(torch.sum(g * v) for (g, v) in zip(grads, vectors))
    # note that grad() sums gradients of prod_sums
    return grad(prod_sums, params)


M = 5
print("V,D,P,N,grad,hvp")
for V in [100_000, 300_000]:
    for D in [100, 300]:
        for N in [100, 500, 1000, 5000, 10_000, 15_000]:
            model = GloVe(V, D)
            if CUDA:
                model.cuda()
            P = count_parameters(model)
            params = tuple(p for p in model.parameters() if p.requires_grad)
            # print(V, D, N, num_params)
            # time gradient
            t0 = time()
            # print("Computing gradient...")
            for i in range(M):
                # print("run:", i)
                loss = model.loss(make_batch(N, V))
                v = grad(loss, params)

            grad_time = (time() - t0) / M
            # print("Gradient time", grad_time)
            # time HVP
            t0 = time()
            loss = model.loss(make_batch(100_000, V))
            v = grad(loss, params)
            for i in range(M):
                # print("run:", i)
                loss = model.loss(make_batch(N, V))
                Hv = hvp(loss, params, v)

            hvp_time = (time() - t0) / M
            # print("HVP time", hvp_time)
            print(",".join([str(x) for x in (V, D, P, N, grad_time, hvp_time)]))
