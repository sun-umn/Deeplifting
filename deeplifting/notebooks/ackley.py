# stdlib
import sys
import time

# Adding PyGRANSO directories. Should be modified by user
sys.path.append("/home/seanschweiger/opt/dl_testing")

# stdlib
import math  # noqa: E402

# third party
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from pygranso.private.getNvar import getNvarTorch  # noqa: E402
from pygranso.pygranso import pygranso  # noqa: E402
from pygranso.pygransoStruct import pygransoStruct  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.fc3_bn = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.fc4_bn = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


model = Net().to(device=device, dtype=torch.double)


def user_fn(model, inputs):
    x1 = model(inputs)[0]
    x2 = model(inputs)[1]
    # obj fn.
    f = (
        -20 * torch.exp(-0.2 * (0.5 * (x1**2 + x2**2)) ** 0.5)
        - torch.exp(0.5 * (torch.cos(2 * math.pi * x1) + torch.cos(2 * math.pi * x2)))
        + 20
        + math.e
    )
    ci = pygransoStruct()
    # constraints -32.678 <= x1 <= 32.678, -32.678 <= x2 <= 32.678
    ci.c1 = -x1 - 32.678
    ci.c2 = x1 - 32.678
    ci.c3 = -x2 - 32.678
    ci.c4 = x2 - 32.678
    ce = None
    return [f, ci, ce]


for i in range(10):
    opts = pygransoStruct()
    opts.torch_device = device
    nvar = getNvarTorch(model.parameters())
    opts.x0 = (
        torch.nn.utils.parameters_to_vector(model.parameters())
        .detach()
        .reshape(nvar, 1)
    )
    # opts.fvalquit = 1e-6ut you can enable them
    opts.print_level = 1
    opts.print_frequency = 10
    # opts.print_ascii = True
    opts.limited_mem_size = 100
    # opts.print_ascii = True
    z = torch.randn(20).to(torch.double)
    comb_fn = lambda model: user_fn(model, z)  # noqa
    start = time.time()
    soln = pygranso(var_spec=model, combined_fn=comb_fn, user_opts=opts)
    end = time.time()
    print("Total Wall Time: {}s".format(end - start))
    print(soln.final.x)
    print(soln.final.x.shape)
