# %% This .py code is used to plot results
# author: Hong-Wei Li, Email: hongweili@seu.edu.cn

import math
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from scipy.io import loadmat

params = {'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times',
          'text.latex.preamble': ''.join([r'\usepackage{fontenc}'
                                          r'\usepackage{newtxtext,newtxmath}'
                                          r'\usepackage{amsmath}'])
          }
plt.rcParams.update(params)
cm = 1 / 2.54

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

file_dir = os.path.join(case_dir, 'Zener_CSNN_2layer1_2state_2neuron1.mat')
f = loadmat(file_dir)
Train_data = Train_data + (f['Train_y'],)  # t u y y
Train_y_l = f['Train_y_l']
Train_y_n = Train_data[3] - Train_y_l

temp1 = Test_data[0] + (f['Test_y_1_4'],)
temp2 = Test_data[1] + (f['Test_y_5_8'],)
Test_data = (temp1, temp2)  # t u y y
Test_y_l = (f['Test_y_l_1_4'], f['Test_y_l_5_8'])
Test_y_n = (Test_data[0][3]-f['Test_y_l_1_4'], Test_data[1][3]-f['Test_y_l_5_8'])
del f

# Calculate the Pearson correlation coefficients
corr = np.zeros((9, 1))
corr[0] = np.corrcoef(Train_data[2], Train_data[3])[0, 1]

u_e = np.zeros((9, 1))
y_e = np.zeros((9, 1))
u_e[0] = np.sqrt(np.square(Train_data[1]).mean())
y_e[0] = np.sqrt(np.square(Train_data[3]).mean())
y_index = np.zeros((9, 1))
temp1 = np.sqrt(np.square(Train_y_l).mean())
temp2 = np.sqrt(np.square(Train_y_n).mean())
y_index[0] = temp2/(temp1+temp2)*100

for i in range(2):
    for j in range(4):
        k = 4 * i + j + 1
        corr[k] = np.corrcoef(Test_data[i][2][j, :], Test_data[i][3][j, :])[0, 1]
        u_e[k] = np.sqrt(np.square(Test_data[i][1][j, :]).mean())
        y_e[k] = np.sqrt(np.square(Test_data[i][3][j, :]).mean())
        temp1 = np.sqrt(np.square(Test_y_l[i][j, :]).mean())
        temp2 = np.sqrt(np.square(Test_y_n[i][j, :]).mean())
        y_index[k] = temp2 / (temp1 + temp2)*100

print("------Pearson correlation coefficients------")
print("Case 1: %.8f" % corr[0])
for i in range(8):
    print("Case %d: %.8f" % (i + 2, corr[i + 1]))

# %% Plot the input
fig1 = plt.figure(figsize = (16 * cm, 8 * cm))
ax = fig1.add_subplot(331)
Train_tend = 10  # the first Train_tend seconds of data are used for training
Train_Nt0 = math.floor(Train_tend / Train_dt) + 1
ax.plot(Train_data[0][0, 0:Train_Nt0], Train_data[1][0, 0:Train_Nt0], color = 'b', lw = 0.5)
ax.plot(Train_data[0][0, Train_Nt0:-1], Train_data[1][0, Train_Nt0:-1], color = 'k', lw = 0.5)
ax.plot([10, 10], [-8, 8], 'r', lw = 1)
ax.text(5, -6, 'Training', ha = 'center', va = 'center', fontsize = 8)

for i in range(2):
    for j in range(4):
        k = 4 * i + j + 2
        ax = fig1.add_subplot(3, 3, k)
        ax.plot(Test_data[i][0][0, :], Test_data[i][1][j, :], color = 'k', lw = 0.5)

ax = plt.subplot(331)
ax.set_ylabel(r'Disp. (mm)', fontsize = 8, labelpad = 1)
ax.set_ylim([-8, 8])
ax.set_yticks(np.arange(-8, 8.1, 4))

ax = plt.subplot(332)
ax.set_ylim([-10, 10])
ax.set_yticks(np.arange(-10, 10.1, 5))

ax = plt.subplot(333)
ax.set_ylim([-4, 4])
ax.set_yticks(np.arange(-4, 4.1, 2))

ax = plt.subplot(334)
ax.set_ylabel(r'Disp. (mm)', fontsize = 8, labelpad = 1)
ax.set_ylim([-2, 2])
ax.set_yticks(np.arange(-2, 2.1, 1))

ax = plt.subplot(335)
ax.set_ylim([-0.8, 0.8])
ax.set_yticks(np.arange(-0.8, 0.81, 0.4))

ax = plt.subplot(336)
ax.set_ylim([-40, 40])
ax.set_yticks(np.arange(-40, 40.1, 20))

ax = plt.subplot(337)
ax.set_ylabel(r'Disp. (mm)', fontsize = 8, labelpad = 1)
ax.set_xlabel(r'Time (s)', fontsize = 8, labelpad = 1)
ax.set_ylim([-20, 20])
ax.set_yticks(np.arange(-20, 20.1, 10))

ax = plt.subplot(338)
ax.set_xlabel(r'Time (s)', fontsize = 8, labelpad = 1)
ax.set_ylim([-10, 10])
ax.set_yticks(np.arange(-10, 10.1, 5))

ax = plt.subplot(339)
ax.set_xlabel(r'Time (s)', fontsize = 8, labelpad = 1)
ax.set_ylim([-6, 6])
ax.set_yticks(np.arange(-6, 6.1, 3))

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    text_box = AnchoredText(r'Case ' + str(i + 1), frameon = True, loc = 'upper center', pad = 0.2,
                            borderpad = 0, prop = dict(fontsize = 8))
    plt.setp(text_box.patch, facecolor = 'white', alpha = 1, lw = .75)
    ax.add_artist(text_box)
    ax.tick_params(axis = 'y', labelsize = 8)
    ax.set_xlim([0, 40])
    ax.set_xticks(np.arange(0, 40.1, 10))
    ax.tick_params(axis = 'x', labelsize = 8)
    ax.tick_params(direction = "in")  # plt.
    plt.grid(lw = 0.5)
    ax.set_axisbelow(True)

fig1.tight_layout(pad = 0.1)
plt.subplots_adjust(hspace = 0.25, wspace = 0.2)
plt.show()

file_dir = os.path.join(case_dir, 'F_Zener_BLWN_u_.pdf')
# fig1.savefig(fname = file_dir, format = "pdf")

# %% Plot results for Case 6
col = ['k', 'r']
gs = gridspec.GridSpec(2, 2)
ax_list = []
fig3 = plt.figure(figsize = (16 * cm, 8 * cm))
ax_list.append(fig3.add_subplot(gs[0, :]))
ax = ax_list[0]
ax.plot(Test_data[1][0][0, :], Test_data[1][2][0, :], color = col[0], lw = 0.5,
        label = 'Ground truth')
ax.plot(Test_data[1][0][0, :], Test_data[1][3][0, :], color = col[1], dashes = [2, 2], lw = 0.5,
        label = 'Prediction')
ax.set_ylabel(r'Force. (kN)', fontsize = 8, labelpad = 1)
ax.set_xlim([0, 100])
ax.set_xticks(np.arange(0, 100.1, 10))
text_box = AnchoredText(r'Case 6', frameon = True, loc = 'upper center', pad = 0.2,
                        borderpad = 0, prop = dict(fontsize = 8))
plt.setp(text_box.patch, facecolor = 'white', alpha = 1, lw = .75)
ax.add_artist(text_box)
legend = ax.legend(loc = 'upper right', bbox_to_anchor = (1., 1.), borderpad = 0.3, borderaxespad = 0,
                   handlelength = 2.8,
                   edgecolor = 'black', fontsize = 8, ncol = 2, columnspacing = 0.5,
                   handletextpad = 0.3)  # labelspacing=0
legend.get_frame().set_boxstyle('Square', pad = 0.0)
legend.get_frame().set_lw(0.75)
legend.get_frame().set_alpha(None)
for legobj in legend.legend_handles:
    legobj.set_lw(0.75)
ax.tick_params(direction = "in")  # plt.
ax.grid(lw = 0.5)

ax_list.append(fig3.add_subplot(gs[1, 0]))
ax = ax_list[1]
ax.plot(Test_data[1][0][0, :], Test_data[1][2][0, :], color = col[0], lw = 0.5,
        label = 'Ground truth')
ax.plot(Test_data[1][0][0, :], Test_data[1][3][0, :], color = col[1], dashes = [2, 2], lw = 0.5,
        label = 'Ground truth')
ax.set_ylabel(r'Force. (kN)', fontsize = 8, labelpad = 1)
ax.set_xlim([6, 10])
ax.set_xticks(np.arange(6, 10.1, 1))
ax.set_xlabel(r'Time (s)', fontsize = 8, labelpad = 1)

ax_list.append(fig3.add_subplot(gs[1, 1]))
ax = ax_list[2]
ax.plot(Test_data[1][0][0, :], Test_data[1][2][0, :], color = col[0], lw = 0.5,
        label = 'Ground truth')
ax.plot(Test_data[1][0][0, :], Test_data[1][3][0, :], color = col[1], dashes = [2, 2], lw = 0.5,
        label = 'Ground truth')
ax.set_xlim([96, 100])
ax.set_xticks(np.arange(96, 100.1, 1))
ax.set_xlabel(r'Time (s)', fontsize = 8, labelpad = 1)

for i in range(0, 3):
    ax = ax_list[i]
    ax.set_ylim([-80, 80])
    ax.set_yticks(np.arange(-80, 80.1, 40))
    ax.tick_params(axis = 'y', labelsize = 8)
    ax.tick_params(axis = 'x', labelsize = 8)
    ax.tick_params(direction = "in")  # plt.
    ax.grid(lw = 0.5)
    ax.set_axisbelow(True)

fig3.tight_layout(pad = 0.1)
plt.subplots_adjust(hspace = 0.2, wspace = 0.12)
plt.show()

file_dir = os.path.join(case_dir, 'F_Zener_BLWN_case6.pdf')
# fig3.savefig(fname = file_dir, format = "pdf")