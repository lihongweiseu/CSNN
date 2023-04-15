# %% This .py code is used to plot results
# author: Hong-Wei Li, Email: hongweili@seu.edu.cn

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.io import loadmat

params = {'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times',
          'text.latex.preamble': ''.join([r'\usepackage{fontenc}'
                                          r'\usepackage{newtxtext,newtxmath}'
                                          r'\usepackage{amsmath}'])
          }
plt.rcParams.update(params)

# Prepare data
dt = 0.05
Nt = 1001
tend = 50
t = np.linspace(0, tend, Nt).reshape(-1, 1)
NTrain = 10  # number of ground motions for training
NTest = 89  # number of ground motions for testing
case_dir = os.path.join('.\Illustrative example 2 Nonlinear system subjected to acceleration excitation')
file_dir = os.path.join(case_dir, 'num_ref.mat')
f = loadmat(file_dir)
Train_u = np.transpose(f['Train_u'])
Train_y_ref = np.transpose(f['Train_y_ref'])*100
Test_u = np.transpose(f['Test_u'])
Test_y_ref = np.transpose(f['Test_y_ref'])*100

file_dir = os.path.join(case_dir, 'num_PhyCNN.mat')
f = loadmat(file_dir)
Train_y_PhyCNN = np.transpose(f['Train_y_PhyCNN'])*100
Test_y_PhyCNN = np.transpose(f['Test_y_PhyCNN'])*100

file_dir = os.path.join(case_dir, 'num_CSNN_2layer1_2state_2neuron1.mat')
f = loadmat(file_dir)
Train_y = np.transpose(f['Train_y'])*100
Test_y = np.transpose(f['Test_y'])*100
Train_y_l = np.transpose(f['Train_y_l'])*100
Test_y_l = np.transpose(f['Test_y_l'])*100
del f

# Calculate the Pearson correlation coefficients
Train_corr_PhyCNN = np.zeros((NTrain, 1))
Train_corr = np.zeros((NTrain, 1))
Train_y_n = Train_y-Train_y_l
Train_u_e = np.zeros((NTrain, 1))
Train_y_e = np.zeros((NTrain, 1))
Train_y_le = np.zeros((NTrain, 1))
Train_y_ne = np.zeros((NTrain, 1))
Train_y_index = np.zeros((NTrain, 1))

Test_corr_PhyCNN = np.zeros((NTest, 1))
Test_corr = np.zeros((NTest, 1))
Test_y_n = Test_y-Test_y_l
Test_u_e = np.zeros((NTest, 1))
Test_y_e = np.zeros((NTest, 1))
Test_y_le = np.zeros((NTest, 1))
Test_y_ne = np.zeros((NTest, 1))
Test_y_index = np.zeros((NTest, 1))


for i in range(NTrain):
    A = Train_y_ref[:, i:i + 1].reshape(1, -1)
    B = Train_y_PhyCNN[:, i:i + 1].reshape(1, -1)
    Train_corr_PhyCNN[i:i + 1, :] = np.corrcoef(A, B)[0, 1]
    B1 = Train_y[:, i:i + 1].reshape(1, -1)
    Train_corr[i:i + 1, :] = np.corrcoef(A, B1)[0, 1]

    Train_u_e[i: i + 1, :] = np.sqrt(np.square(Train_u[:, i:i + 1]).mean())
    Train_y_e[i: i + 1, :] = np.sqrt(np.square(Train_y[:, i:i + 1]).mean())
    Train_y_le[i: i + 1, :] = np.sqrt(np.square(Train_y_l[:, i:i + 1]).mean())
    Train_y_ne[i: i + 1, :] = np.sqrt(np.square(Train_y_n[:, i:i + 1]).mean())
    Train_y_index[i: i + 1, :] = Train_y_ne[i: i + 1, :] / (Train_y_le[i: i + 1, :] + Train_y_ne[i: i + 1, :])*100

for i in range(NTest):
    A = Test_y_ref[:, i:i + 1].reshape(1, -1)
    B = Test_y_PhyCNN[:, i:i + 1].reshape(1, -1)
    Test_corr_PhyCNN[i:i + 1, :] = np.corrcoef(A, B)[0, 1]
    B = Test_y[:, i:i + 1].reshape(1, -1)
    Test_corr[i:i + 1, :] = np.corrcoef(A, B)[0, 1]

    Test_u_e[i: i + 1, :] = np.sqrt(np.square(Test_u[:, i:i + 1]).mean())
    Test_y_e[i: i + 1, :] = np.sqrt(np.square(Test_y[:, i:i + 1]).mean())
    Test_y_le[i: i + 1, :] = np.sqrt(np.square(Test_y_l[:, i:i + 1]).mean())
    Test_y_ne[i: i + 1, :] = np.sqrt(np.square(Test_y_n[:, i:i + 1]).mean())
    Test_y_index[i: i + 1, :] = Test_y_ne[i: i + 1, :] / (Test_y_le[i: i + 1, :] + Test_y_ne[i: i + 1, :])*100

corr_PhyCNN = np.vstack((Train_corr_PhyCNN, Test_corr_PhyCNN))
corr = np.vstack((Train_corr, Test_corr))
y_e = np.vstack((Train_y_e, Test_y_e))
y_index = np.vstack((Train_y_index, Test_y_index))
cm = 1 / 2.54
N = np.linspace(1, NTrain + NTest, NTrain + NTest).reshape(-1, 1)

# %% Plot the Pearson correlation coefficients over datasets
fig = plt.figure(figsize=(12 * cm, 5 * cm))
ax = fig.add_subplot(111)
ax.plot([NTrain+0.5, NTrain+0.5], [0.5, 1], 'k', lw=1)
ax.scatter(N, corr_PhyCNN, c='b', marker='o', s=30, alpha=0.8, linewidths=0, label='PhyCNN') #
ax.scatter(N, corr, c='r', marker='^', s=30, alpha=0.8, linewidths=0, label='CSNN')
ax.set_ylim([0.5, 1])
ax.set_yticks(np.arange(0.5, 1.01, 0.1))
ax.tick_params(axis='y', labelsize=8)
ax.set_ylabel(r'Pearson corr.', fontsize=8, labelpad=1)
ax.set_xlim([0, 100])
ax.set_xticks(np.arange(0, 100.1, 10.0))
ax.tick_params(axis='x', labelsize=8)
ax.set_xlabel(r'Case no.', fontsize=8, labelpad=1)
ax.text(5, 0.55, 'Training', ha='center', va='center', fontsize=8)
ax.text(35, 0.55, 'Testing', ha='center', va='center', fontsize=8)
legend = ax.legend(loc='lower right', bbox_to_anchor=(0.9, 0.08), borderpad=0.3, borderaxespad=0, handlelength=2.8,
                   edgecolor='black', fontsize=8, ncol=1, columnspacing=0.5, handletextpad=0.0)  # labelspacing=0
legend.get_frame().set_boxstyle('Square', pad=0.0)
legend.get_frame().set_lw(0.75)
legend.get_frame().set_alpha(None)
for obj in legend.legend_handles:
    obj.set_lw(0)
ax.tick_params(direction="in")  # plt.
ax.grid(lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.1)
plt.show()

file_dir = os.path.join(case_dir, 'F_num_corr.pdf')
# fig.savefig(fname = file_dir, format = "pdf")

# %% Plot the time histories for datasets 30 65
col = ['k', 'b', 'r']
c_dash = [[8, 4], [2, 2]]
fig = plt.figure(figsize=(16 * cm, 8 * cm))
ax = fig.add_subplot(211)
ax.plot(t, Test_y_ref[:, 19:20], color=col[0], lw=0.5, label='Ground truth')
ax.plot(t, Test_y_PhyCNN[:, 19:20], color=col[1], dashes=[8, 4], lw=0.5, label='PhyCNN')
ax.plot(t, Test_y[:, 19:20], color=col[2], dashes=[2, 2], lw=0.5, label='CSNN')
ax.set_ylim([-40, 40])
ax.set_yticks(np.arange(-40, 40.1, 20))
ax.text(42.5, -30, 'Case 30', ha='center', va='center', fontsize=8)
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderpad=0.3, borderaxespad=0, handlelength=2.8,
                   edgecolor='black', fontsize=8, ncol=3, columnspacing=0.5, handletextpad=0.3)  # labelspacing=0
legend.get_frame().set_boxstyle('Square', pad=0.0)
legend.get_frame().set_lw(0.75)
legend.get_frame().set_alpha(None)
for obj in legend.legend_handles:
    obj.set_lw(0.75)

ax = fig.add_subplot(212)
ax.plot(t, Test_y_ref[:, 83:84], color=col[0], lw=0.5)
ax.plot(t, Test_y_PhyCNN[:, 83:84], color=col[1], dashes=c_dash[0], lw=0.5)
ax.plot(t, Test_y[:, 83:84], color=col[2], dashes=c_dash[1], lw=0.5)
ax.set_ylim([-20, 20])
ax.set_yticks(np.arange(-20, 20.1, 10))
ax.set_xlabel(r'Time (s)', fontsize=8, labelpad=1)
ax.text(42.5, -15, 'Case 94', ha='center', va='center', fontsize=8)

for i in range(2):
    ax = plt.subplot(2, 1, i + 1)
    ax.tick_params(axis = 'y', labelsize=8)
    ax.set_xlim([0, 50])
    ax.set_xticks(np.arange(0, 50.1, 5))
    ax.tick_params(axis = 'x', labelsize=8)
    ax.set_ylabel(r'Disp. (cm)', fontsize=8, labelpad = 1)
    ax.tick_params(direction="in")  # plt.
    plt.grid(lw=0.5)
    ax.set_axisbelow(True)

fig.tight_layout(pad=0.1)
plt.subplots_adjust(hspace=0.25, wspace=0.15)
plt.show()
file_dir = os.path.join(case_dir, 'F_num_dis.pdf')
# fig.savefig(fname = file_dir, format = "pdf")

# %% Plot the Pearson correlation coefficients over datasets
fig = plt.figure(figsize=(12 * cm, 5 * cm))
ax = fig.add_subplot(111)
# ax.scatter(N, y_e, c='b', marker='o', s=50, alpha=0.8, linewidths=0, label='PhyCNN')
ax.scatter(Test_y_e, Test_y_index, c='g', marker='D', s=20, alpha=0.8, linewidths=0, label='Testing case')
ax.scatter(Train_y_e, Train_y_index, c='m', marker='*', s=60, alpha=0.8, linewidths=0, label='Training case')
ax.set_ylim([40, 90])
ax.set_yticks(np.arange(40, 91, 10))
ax.tick_params(axis='y', labelsize=8)
ax.set_ylabel(r'Nonlinear ratio (\%)', fontsize=8, labelpad=1)
ax.set_xlim([0, 14])
ax.set_xticks(np.arange(0, 14.1, 2))
ax.tick_params(axis='x', labelsize=8)
ax.set_xlabel('Predictions\''r' RMS ($\rm cm/s^2$)', fontsize=8, labelpad=1)

handles,labels = ax.get_legend_handles_labels()
handles = [handles[1], handles[0]]
labels = [labels[1], labels[0]]
legend = ax.legend(handles,labels,loc='lower right', bbox_to_anchor=(0.95, 0.08), borderpad=0.3, borderaxespad=0, handlelength=2.8,
                   edgecolor='black', fontsize=8, ncol=1, columnspacing=0.5, handletextpad=0.0)  # labelspacing=0
legend.get_frame().set_boxstyle('Square', pad=0.0)
legend.get_frame().set_lw(0.75)
legend.get_frame().set_alpha(None)
for obj in legend.legend_handles:
    obj.set_lw(0.0)
ax.tick_params(direction="in")  # plt.
ax.grid(lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.1)
plt.show()
file_dir = os.path.join(case_dir, 'F_num_Rn.pdf')
# fig.savefig(fname = file_dir, format = "pdf")