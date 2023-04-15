# %% This code is to test CSNN by using Zener model with BLWN inputs
# author: Hong-Wei Li, Email: hongweili@seu.edu.cn

import math
import random
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch import nn, optim
from scipy.io import savemat, loadmat

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

# Define the configuration of CSNN
input_size = 1
output_size = 1
state_size = 2  # user defined

state_layer_non = 2  # number of nonlinear layers for state derivatives
state_neuron_non = np.zeros(state_layer_non, dtype=np.int32)  # size of each nonlinear layer
for i in range(state_layer_non):
    state_neuron_non[i] = state_size  # user defined

output_layer_non = 1  # number of nonlinear layers for outputs
output_neuron_non = np.zeros(output_layer_non, dtype=np.int32)  # size of each nonlinear layer
for i in range(output_layer_non):
    output_neuron_non[i] = output_size  # user defined


# Define CSNN
class CSNN(nn.Module):
    def __init__(self):
        super(CSNN, self).__init__()

        layer_non = [nn.Linear(input_size + state_size, state_neuron_non[0], bias=False), torch.nn.Tanh()]
        if state_layer_non > 1:
            for ii in range(state_layer_non - 1):
                layer_non.append(nn.Linear(state_neuron_non[ii], state_neuron_non[ii + 1], bias=False))
                layer_non.append(torch.nn.Tanh())
        layer_non.append(nn.Linear(state_neuron_non[-1], state_size, bias=False))
        self.StateNet_non = nn.Sequential(*layer_non)
        self.StateNet_lin = nn.Linear(input_size + state_size, state_size, bias=False)

        layer_non = [nn.Linear(input_size + state_size, output_neuron_non[0], bias=False), torch.nn.Tanh()]
        if output_layer_non > 1:
            for ii in range(output_layer_non - 1):
                layer_non.append(nn.Linear(output_neuron_non[ii], output_neuron_non[ii + 1], bias=False))
                layer_non.append(torch.nn.Tanh())
        layer_non.append(nn.Linear(output_neuron_non[-1], output_size, bias=False))
        self.OutputNet_non = nn.Sequential(*layer_non)
        self.OutputNet_lin = nn.Linear(input_size + state_size, output_size, bias=False)

    def forward(self, input_state):
        state_d_non = self.StateNet_non(input_state)
        state_d_lin = self.StateNet_lin(input_state)
        state_d = state_d_non + state_d_lin
        output_non = self.OutputNet_non(input_state)
        output_lin = self.OutputNet_lin(input_state)
        output = output_non + output_lin
        output_state_d = torch.cat((output, state_d), dim=1)
        return output_state_d


# Create model
CSNN_model = CSNN()
criterion = nn.MSELoss()

# Prepare data
case_dir = os.path.join('.\Illustrative example 1 Shear Zener model of viscoelastic dampers')
file_dir = os.path.join(case_dir, 'Zener_data.mat')
f = loadmat(file_dir)
Train_dt = f['Train_dt'][0, 0]
Train_Nt = f['Train_Nt'][0, 0]
Train_data = f['Train_data'].item()  # t u y

Test_dt = f['Test_dt'][:, 0]
Test_Nt = f['Test_Nt'][:, 0]
Test_data = f['Test_data'][:, 0]
Test_data = (Test_data[0].item(), Test_data[1].item())
del f

Train_u = torch.tensor(Train_data[1], dtype=torch.float)
Train_y_ref = torch.tensor(Train_data[2], dtype=torch.float).to(device)
# %% Training
optimizer = optim.Adam(CSNN_model.parameters(), 0.01)
training_num = 6000
Train_tend = 10  # the first Train_tend seconds of data are used for training
Train_Nt0 = math.floor(Train_tend / Train_dt) + 1
loss_all = np.zeros((training_num + 1, 1))  # store loss values
start = time.time()
for counter in range(training_num + 1):
    loss = torch.tensor([[0]], dtype=torch.float).to(device)
    x = torch.zeros(1, state_size)
    for i in range(Train_Nt0 - 1):
        u = Train_u[:, i:i + 1]
        x0 = x
        u_x = torch.cat((u, x), dim=1)
        y = CSNN_model(u_x)[:, 0:output_size].to(device)
        loss += criterion(y, Train_y_ref[:, i:i + 1])
        k1 = CSNN_model(u_x)[:, output_size:]

        u = (u + Train_u[:, i + 1:i + 2]) / 2
        x = x0 + k1 * Train_dt / 2
        u_x = torch.cat((u, x), dim=1)
        k2 = CSNN_model(u_x)[:, output_size:]

        x = x0 + k2 * Train_dt / 2
        u_x = torch.cat((u, x), dim=1)
        k3 = CSNN_model(u_x)[:, output_size:]

        u = Train_u[:, i + 1:i + 2]
        x = x0 + k3 * Train_dt
        u_x = torch.cat((u, x), dim=1)
        k4 = CSNN_model(u_x)[:, output_size:]

        x = x0 + (k1 + 2 * k2 + 2 * k3 + k4) * Train_dt / 6

    u_x = torch.cat((u, x), dim=1)
    y = CSNN_model(u_x)[:, 0:output_size].to(device)
    loss += criterion(y, Train_y_ref[:, i + 1:i + 2])
    loss_all[counter:counter + 1, :] = loss.item()
    if counter < training_num:
        CSNN_model.zero_grad()
        loss.backward()
        optimizer.step()

    if (counter + 1) % 10 == 0 or counter == 0:
        print(f"iteration: {counter + 1}, loss: {loss.item()}")
        end = time.time()
        per_time = (end - start) / (counter + 1)
        print("Average training time: %.3f s per training" % per_time)
        print("Cumulative training time: %.3f s" % (end - start))
        left_time = (training_num - counter + 1) * per_time
        print(f"Executed at {time.strftime('%H:%M:%S', time.localtime())},", "left time: %.3f s\n" % left_time)

end = time.time()
print("Total training time: %.3f s" % (end - start))
print(f"loss: {loss.item()}")

# # Save model
# torch.save(CSNN_model.state_dict(), "Zener_CSNN_2layer1_2state_2neuron1_6000_10s_loss2834em1.pt")

# %%
file_dir = os.path.join(case_dir, 'Zener_CSNN_2layer1_2state_2neuron1_6000_10s_loss2834em1.pt')
CSNN_model.load_state_dict(torch.load(file_dir))
CSNN_model_l = CSNN()
CSNN_model_l.load_state_dict(torch.load(file_dir))
CSNN_model_l.StateNet_non[0].weight.data = torch.zeros(state_neuron_non[0], input_size+state_size)
CSNN_model_l.OutputNet_non[0].weight.data = torch.zeros(output_neuron_non[0], input_size+state_size)

# %% Plot training result
Train_y = np.zeros((1, Train_Nt))
x = torch.zeros(1, state_size)
Train_y_l = np.zeros((1, Train_Nt))
x_l = torch.zeros(1, state_size)
for i in range(Train_Nt - 1):
    u = Train_u[:, i:i + 1]
    x0 = x
    u_x = torch.cat((u, x), dim=1)
    y = CSNN_model(u_x)[:, 0:output_size]
    Train_y[:, i:i + 1] = y.detach().numpy()
    k1 = CSNN_model(u_x)[:, output_size:]
    x0_l = x_l
    u_x_l = torch.cat((u, x_l), dim = 1)
    y_l = CSNN_model_l(u_x_l)[:, 0:output_size]
    Train_y_l[:, i:i + 1] = y_l.detach().numpy()
    k1_l = CSNN_model_l(u_x)[:, output_size:]

    u = (u + Train_u[:, i + 1:i + 2]) / 2
    x = x0 + k1 * Train_dt / 2
    u_x = torch.cat((u, x), dim=1)
    k2 = CSNN_model(u_x)[:, output_size:]
    k2_l = CSNN_model_l(u_x)[:, output_size:]

    x = x0 + k2 * Train_dt / 2
    u_x = torch.cat((u, x), dim=1)
    k3 = CSNN_model(u_x)[:, output_size:]
    k3_l = CSNN_model_l(u_x)[:, output_size:]

    u = Train_u[:, i + 1:i + 2]
    x = x0 + k3 * Train_dt
    u_x = torch.cat((u, x), dim=1)
    k4 = CSNN_model(u_x)[:, output_size:]
    k4_l = CSNN_model_l(u_x)[:, output_size:]

    x = x0 + (k1 + 2 * k2 + 2 * k3 + k4) * Train_dt / 6
    x_l = x0_l + (k1_l + 2 * k2_l + 2 * k3_l + k4_l) * Train_dt / 6

u_x = torch.cat((u, x), dim=1)
y = CSNN_model(u_x)[:, 0:output_size]
Train_y[:, Train_Nt - 1:Train_Nt] = y.detach().numpy()
u_x_l = torch.cat((u, x_l), dim=1)
y_l = CSNN_model_l(u_x_l)[:, 0:output_size]
Train_y_l[:, Train_Nt - 1:Train_Nt] = y_l.detach().numpy()
Train_y_n = Train_y-Train_y_l

plt.plot(Train_data[0].reshape(-1, 1), Train_y_ref.reshape(-1, 1))
plt.plot(Train_data[0].reshape(-1, 1), Train_y.reshape(-1, 1))
plt.show()

# %% Plot testing result
Test_y = (np.zeros((4, Test_Nt[0])), np.zeros((4, Test_Nt[1])))
Test_y_l = (np.zeros((4, Test_Nt[0])), np.zeros((4, Test_Nt[1])))
Test_y_n = (np.zeros((4, Test_Nt[0])), np.zeros((4, Test_Nt[1])))
for j in range(2):
    x = torch.zeros(4, state_size)
    x_l = torch.zeros(4, state_size)
    Test_u = torch.tensor(Test_data[j][1], dtype=torch.float)
    for i in range(Test_Nt[j] - 1):
        u = Test_u[:, i:i + 1]
        x0 = x
        u_x = torch.cat((u, x), dim = 1)
        y = CSNN_model(u_x)[:, 0:output_size]
        Test_y[j][:, i:i + 1] = y.detach().numpy()
        k1 = CSNN_model(u_x)[:, output_size:]
        x0_l = x_l
        u_x_l = torch.cat((u, x_l), dim = 1)
        y_l = CSNN_model_l(u_x_l)[:, 0:output_size]
        Test_y_l[j][:, i:i + 1] = y_l.detach().numpy()
        k1_l = CSNN_model_l(u_x)[:, output_size:]

        u = (u + Test_u[:, i + 1:i + 2]) / 2
        x = x0 + k1 * Test_dt[j] / 2
        u_x = torch.cat((u, x), dim = 1)
        k2 = CSNN_model(u_x)[:, output_size:]
        k2_l = CSNN_model_l(u_x)[:, output_size:]

        x = x0 + k2 * Test_dt[j] / 2
        u_x = torch.cat((u, x), dim = 1)
        k3 = CSNN_model(u_x)[:, output_size:]
        k3_l = CSNN_model_l(u_x)[:, output_size:]

        u = Test_u[:, i + 1:i + 2]
        x = x0 + k3 * Test_dt[j]
        u_x = torch.cat((u, x), dim = 1)
        k4 = CSNN_model(u_x)[:, output_size:]
        k4_l = CSNN_model_l(u_x)[:, output_size:]

        x = x0 + (k1 + 2 * k2 + 2 * k3 + k4) * Test_dt[j] / 6
        x_l = x0 + (k1_l + 2 * k2_l + 2 * k3_l + k4_l) * Test_dt[j] / 6

    u_x = torch.cat((u, x), dim=1)
    y = CSNN_model(u_x)[:, 0:output_size]
    Test_y[j][:, Test_Nt[j] - 1:Test_Nt[j]] = y.detach().numpy()
    u_x_l = torch.cat((u, x_l), dim = 1)
    y_l = CSNN_model_l(u_x_l)[:, 0:output_size]
    Test_y_l[j][:, Test_Nt[j] - 1:Test_Nt[j]] = y_l.detach().numpy()
    Test_y_n[j][:, :] = Test_y[j][:, :] - Test_y_l[j][:, :]

j = 1
i = 0
plt.plot(Test_data[j][0][0:1, :].reshape(-1, 1), Test_data[j][2][i:i+1, :].reshape(-1, 1))
plt.plot(Test_data[j][0][0:1, :].reshape(-1, 1), Test_y[j][i:i+1, :].reshape(-1, 1))
plt.show()

# %% save data
file_dir = os.path.join(case_dir, 'Zener_CSNN_2layer1_2state_2neuron1.mat')
savemat(file_dir, {'Train_y': Train_y, 'Train_y_l': Train_y_l, 'Test_y_1_4': Test_y[0],'Test_y_5_8': Test_y[1],
                   'Test_y_l_1_4': Test_y_l[0],'Test_y_l_5_8': Test_y_l[1]})

