# Codes for the paper "Modeling of forced-vibration systems using continuous-time state-space neural network"
---  Coded in the Python environment utilizing the powerful deep learning library PyTorch.
## Illustrative example 1 Shear Zener model of viscoelastic dampers
1. _Zener_Sin_RNN_main.py_ is the main Python file for the quick demonstration of the RNN model.
2. _Zener_CSNN_main.py_ is the main Python file for the SINN model.
3. _Zener_preparation.m_ is the Matlab file to generate the banded limited white noises as input signals.
4. _Zener_data.mat_ is the generated banded limited white noises.
5. _Zener_CSNN_2layer1_2state_2neuron1_6000_10s_loss2834em1.pt_ is the trained library of CSNN model, users could load it in _Zener_CSNN_main.py_ to genetate _Zener_CSNN_2layer1_2state_2neuron1.mat_ quickly.
6. _Zener_CSNN_2layer1_2state_2neuron1.mat_ stores the simulation results.
7. _Zener_CSNN_plot.py_ plots the results.

## Illustrative example 2 Nonlinear system subjected to acceleration excitation
1. _num_CSNN_main.py_ is the main Python file for the CSNN model.
2. _num_CSNN_2layer1_2state_2neuron1_3000_loss5316em2.pt_ is the trained library of CSNN model, users could load it in _num_main.py_ to genetate _num_data_pred1_2layer1_2state_2neuron1.mat_ quickly.
3. _num_preparation.py_ prepares the data.
4. _results_num_ag2u.mat_ is the data file coming from the PhyCNN paper "Physics-guided convolutional neural network (PhyCNN) for data-driven seismic response modeling".
5. _numa_ref.mat_ stores the reference results.
6. _num_PhyCNN.mat_ stores the prediction results using the PhyCNN model.
7. _num_CSNN_2layer1_2state_2neuron1.mat_ stores the prediction results using the CSNN model.
8. _num_CSNN_plot.py_ plots the results.

## Illustrative example 3 a 6-Story hotel building with recorded seismic responses
1. _exp_CSNN_main.py_ is the main Python file for the CSNN model.
2. _exp_CSNN_1layer_3state_3neuron1_acc_roof_10000_loss9413ep3.pt_ is the trained library of CSNN model, users could load it in _exp_CSNN_main.py_ to genetate _exp_CSNN_1layer_3state_3neuron1.mat_ quickly.
3. _results_exp_ag2utt.mat_ is the data file coming from the PhyCNN paper
4. _exp_ref.mat_ stores the reference results.
5. _exp_PhyCNN.mat_ stores the prediction results using the PhyCNN model.
6. _exp_CSNN_1layer_3state_3neuron1.mat_ stores the prediction results using the CSNN model.
7. _exp_CSNN_plot.py_ plots the results.
