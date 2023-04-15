# %% This code is to test RNN by using Zener model with a sinusoidal input
# author: Hong-Wei Li, Email: hongweili@seu.edu.cn

import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

# chose to use gpu or cpu
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# To guarantee same results for every running, which might slow down the training speed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)

# Define the configuration of
hidden_size = 10
input_size = 1
output_size = 1
num_layers = 1


# Define RNN
class Rnn(nn.Module):
    def __init__(self):
        super(Rnn, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, u):
        h0 = torch.zeros(num_layers, hidden_size)
        y, hn = self.rnn(u, h0)
        y = self.linear(y)
        return y


# Create model
RNN_model = Rnn()
criterion = nn.MSELoss()

# Prepare data
tend = 30  # total time, must be integer times of dt0
dt0 = 0.04
Nt0 = math.floor(tend / dt0) + 1
t0 = np.linspace(0, tend, Nt0).reshape(-1, 1)
u0 = 8 * np.sin(0.4 * math.pi * t0).reshape(-1, 1)
# y_ref = u0 + 1.6 * math.pi * np.cos(0.4 * math.pi * t0).reshape(-1, 1)

# solving the system using the 4th-order Runge Kutta (RK4) method
a0 = 21.245967741935484
b1 = 2.874
C = -55.509919354838715
y_ref = np.zeros((Nt0, 1))
state = np.zeros((Nt0, 1))
for i in range(Nt0 - 1):
    x1 = state[i:i + 1, :]
    k1 = -a0 * x1 + u0[i:i + 1, :]

    x2 = x1 + k1 * dt0 / 2
    u2 = (u0[i:i + 1, :] + u0[i + 1:i + 2, :]) / 2
    k2 = -a0 * x2 + u2

    x3 = x1 + k2 * dt0 / 2
    k3 = -a0 * x3 + u2

    x4 = x1 + k3 * dt0
    k4 = -a0 * x4 + u0[i + 1:i + 2, :]
    state[i + 1:i + 2, :] = state[i:i + 1, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt0
    y_ref[i + 1:i + 2, :] = C * state[i + 1:i + 2, :] + b1 * u0[i + 1:i + 2, :]

# %% Training
tend_train = 10  # training time, must be integer times of dt0 and not greater than tend
Nt0_train = math.floor(tend_train / dt0) + 1
t0_train = t0[0:Nt0_train, :]
u0_train = torch.tensor(u0[0:Nt0_train, :], dtype = torch.float)
y0_train_ref = torch.tensor(y_ref[0:Nt0_train, :], dtype = torch.float).to(device)

training_num = 3000
RNN_optimizer = optim.Adam(RNN_model.parameters(), 0.01)
RNN_loss_all = np.zeros((training_num + 1, 1))
start = time.time()
for i in range(training_num):
    RNN_y0_train_pred = RNN_model(u0_train).to(device)
    loss = criterion(RNN_y0_train_pred, y0_train_ref)
    RNN_loss_all[i:i + 1, :] = loss.item()
    RNN_model.zero_grad()
    loss.backward()
    RNN_optimizer.step()

    if (i + 1) % 100 == 0 or i == 0:
        print(f"iteration: {i + 1}, RNN loss: {RNN_loss_all[i:i + 1, :].item()}")
        end = time.time()
        per_time = (end - start) / (i + 1)
        print("Average training time: %.6f s per two trainings" % per_time)
        print("Cumulative training time: %.6f s" % (end - start))
        left_time = (training_num - i + 1) * per_time
        print(f"Executed at {time.strftime('%H:%M:%S', time.localtime())},", "left time: %.6f s\n" % left_time)

i = training_num
RNN_y0_train_pred = RNN_model(u0_train).to(device)
loss = criterion(RNN_y0_train_pred, y0_train_ref)
RNN_loss_all[i:i + 1, :] = loss.item()

end = time.time()
print("Total training time: %.3f s" % (end - start))
print(f"RNN loss: {RNN_loss_all[i:i + 1, :].item()}")
# %% Calculate predictions
RNN_y0_pred = RNN_model(torch.tensor(u0, dtype = torch.float))

dt1 = 0.01
Nt1 = math.floor(tend / dt1) + 1
t1 = np.linspace(0, tend, Nt1).reshape(-1, 1)
u1 = 8 * np.sin(0.4 * math.pi * t1).reshape(-1, 1)
RNN_y1_pred = RNN_model(torch.tensor(u1, dtype = torch.float))

dt2 = 0.1
Nt2 = math.floor(tend / dt2) + 1
t2 = np.linspace(0, tend, Nt2).reshape(-1, 1)
u2 = 8 * np.sin(0.4 * math.pi * t2).reshape(-1, 1)
u2_tensor = torch.tensor(u2, dtype = torch.float)
RNN_y2_pred = RNN_model(u2_tensor)

# %% plot predictions with same sampling frequency
params = {'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times',
          'text.latex.preamble': ''.join([r'\usepackage{fontenc}'
                                          r'\usepackage{newtxtext,newtxmath}'
                                          r'\usepackage{amsmath}'])
          }
plt.rcParams.update(params)
cm = 1 / 2.54

fig = plt.figure(figsize = (16 * cm, 5 * cm))
ax = fig.add_subplot(111)
ax.plot(t0, y_ref, color = 'k', lw = 1, label = 'Ground truth')
ax.plot(t0, RNN_y0_pred.detach(), color = 'c', dashes = [8, 4], lw = 1, label = 'RNN ($\Delta t=0.04\,{\\rm s}$)')
ax.plot(t1, RNN_y1_pred.detach(), color = 'r', dashes = [9, 4, 2, 4], lw = 1, label = 'RNN ($\Delta t=0.01\,{\\rm s})$')
ax.plot(t2, RNN_y2_pred.detach(), color = 'b', dashes = [2, 2], lw = 1, label = 'RNN ($\Delta t=0.1\,{\\rm s}$)')
ax.set_ylim([-4, 4])
ax.set_yticks(np.arange(-4, 4.1, 2))
ax.tick_params(axis = 'y', labelsize = 8)
ax.set_ylabel(r'Force. (kN)', fontsize = 8, labelpad = 1)
ax.set_xlim([0, 30])
ax.set_xticks(np.arange(0, 30.1, 5))
ax.tick_params(axis = 'x', labelsize = 8)
ax.set_xlabel(r'Time (s)', fontsize = 8, labelpad = 1)
ax.tick_params(direction = "in")  # plt.
plt.grid(lw = 1)
ax.set_axisbelow(True)
legend = ax.legend(loc = 'upper right', bbox_to_anchor = (1.0, 1.0), borderpad = 0.3, borderaxespad = 0,
                   handlelength = 2.8,
                   edgecolor = 'black', fontsize = 8, ncol = 4, columnspacing = 0.8,
                   handletextpad = 0.3)  # labelspacing=0
legend.get_frame().set_boxstyle('Square', pad = 0.0)
legend.get_frame().set_lw(0.75)
legend.get_frame().set_alpha(None)
for obj in legend.legend_handles:
    obj.set_lw(0.75)

fig.tight_layout(pad = 0.1)
# plt.subplots_adjust(hspace = 0.2, wspace = 0.15)
plt.show()

# fig.savefig(fname = "F_Zener_Sin_RNN.pdf", format = "pdf")
