# %% This .py code is used to check the results (Sec 3.1) of Ref:
# Physics-guided convolutional neural network (PhyCNN) for data-driven seismic resp. modeling
# and prepare the data for our case study
# author: Hong-Wei Li, Email: hongweili@seu.edu.cn
# results_num_ag2u.mat is coming from the Ref code package:
# https://github.com/zhry10/PhyCNN

# Users can run this code to generate num_data_ref.mat and num_data_pred0.mat (already included in the code package)
# or just skip this code and just run the main code num_CSNN_main.py

import matplotlib.gridspec as gridspec
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat, loadmat

# Load data
case_dir = os.path.join('.\Illustrative example 2 Nonlinear system subjected to acceleration excitation')
file_dir = os.path.join(case_dir, 'results_num_ag2u.mat')
f = loadmat(file_dir)
Train_u = f['X_train']  # Training: input data (ground acc.)
Train_y_ref = f['y_train_ref']  # Training: output data reference (disp. resp.)
Train_y_PhyCNN = f['y_train_pred']  # Training: output data prediction (disp. resp.)
# loss0 = f['train_loss'].reshape(-1, 1)
Test_u = f['X_pred']  # Testing: input data (ground acc.)
Test_y_ref = f['y_pred_ref']  # Testing: output data reference (disp. resp.)
Test_y_PhyCNN = f['y_pred']  # Testing: output data prediction (disp. resp.)
# The number 0 at the end of a variable means it is from Ref
# The number 1 at the end of a variable means it is from our case study
t = f['time']  # time vector
del f

# Combine the input and output (reference) datasets
Nt = 1001  # number of data points
u = np.vstack((Train_u, Test_u))
y_ref = np.vstack((Train_y_ref, Test_y_ref))
y = np.zeros((100, Nt))  # y is used to compared with y_ref to check if Ref results are correct
dy = np.zeros((100, Nt))
y_max_err = np.zeros((100, 1))  # maximum errors between y_ref and y
A = np.array([[0, 1], [-20, -1]])
dt = 0.05

# solving the system using the 4th-order Explicit Runge Kutta (RK4) method
for i in range(Nt - 1):
    f1 = np.transpose(np.hstack((y[:, i:i + 1], dy[:, i:i + 1])))
    temp1 = np.vstack([np.zeros((1, 100)), -200 * f1[0:1, :] ** 3])
    temp2 = np.vstack([np.zeros((1, 100)), - np.transpose((u[:, i:i + 1]))])
    k1 = np.dot(A, f1) + temp1 + temp2

    f2 = f1 + k1 * dt / 2
    temp1 = np.vstack([np.zeros((1, 100)), -200 * f2[0:1, :] ** 3])
    temp2 = np.vstack([np.zeros((1, 100)), - np.transpose((u[:, i:i + 1] + u[:, i + 1:i + 2]) / 2)])
    k2 = np.dot(A, f2) + temp1 + temp2

    f3 = f1 + k2 * dt / 2
    temp1 = np.vstack([np.zeros((1, 100)), -200 * f3[0:1, :] ** 3])
    k3 = np.dot(A, f3) + temp1 + temp2

    f4 = f1 + k3 * dt
    temp1 = np.vstack([np.zeros((1, 100)), -200 * f4[0:1, :] ** 3])
    temp2 = np.vstack([np.zeros((1, 100)), - np.transpose((u[:, i + 1:i + 2]))])
    k4 = np.dot(A, f4) + temp1 + temp2

    temp = f1 + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt
    y[:, i + 1:i + 2] = np.transpose(temp[0:1, :])
    dy[:, i + 1:i + 2] = np.transpose(temp[1:2, :])

y_max_err = np.abs(y_ref - y).max(axis = 1).reshape(-1, 1)

# # By checking y_max_err, we found y_ref0[29,:] from Ref is not correct.
# # Run following codes to see details
N = np.linspace(1, 100, 100).reshape(-1, 1)
N1 = np.delete(N, 29, axis = 0)
y_max_err1 = np.delete(y_max_err, 29, axis = 0)

gs = gridspec.GridSpec(2, 2)
ax_list = []
fig = plt.figure()
ax_list.append(fig.add_subplot(gs[0, 0]))
ax = ax_list[0]
ax.plot(N, y_max_err)

ax_list.append(fig.add_subplot(gs[0, 1]))
ax = ax_list[1]
ax.plot(N1, y_max_err1)
ax.set_ylim([0, 6E-16])

ax_list.append(fig.add_subplot(gs[1, :]))
ax = ax_list[2]
ax.plot(t.reshape(-1, 1), y_ref[29:30, :].reshape(-1, 1))
ax.plot(t.reshape(-1, 1), y[29:30, :].reshape(-1, 1))
plt.show()

# y_ref0[29,:] is no. 20 in testing datasets, and it is deleted in our case study
Test_u = np.delete(Test_u, 19, axis = 0)
Test_y_ref = np.delete(Test_y_ref, 19, axis = 0)
Test_y_PhyCNN = np.delete(Test_y_PhyCNN, 19, axis = 0)

# %% save data
file_dir1 = os.path.join(case_dir, 'num_ref.mat')
file_dir2 = os.path.join(case_dir, 'num_PhyCNN.mat')
savemat(file_dir1, {'Train_u': Train_u, 'Train_y_ref': Train_y_ref, 'Test_u': Test_u, 'Test_y_ref': Test_y_ref})
savemat(file_dir2, {'Train_y_PhyCNN': Train_y_PhyCNN, 'Test_y_PhyCNN': Test_y_PhyCNN})
