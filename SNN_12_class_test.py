"""
Test Convolutional SNN
Gesture recognition using 8GHz Radar
Author: Ali Safa - IMEC- KU Leuven
"""

import numpy as np
from eNetworks import mini_eCNN
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.utils import shuffle
#torch.manual_seed(1)
x_sz = 32 #100 #100
y_sz = 32 #48 50

Thr_bits = 6

def percision_transfer(x, precision_bit=16, threshold=1.0):
# =============================================================================
#     if len(x.shape) == 1:
#         o = x.shape
#     else:
#         i,o = x.shape
# =============================================================================
    x_list = x.reshape((-1,))
    
    # chip parameter scaling with threshold normalized to 1
    #W = 4 = precision_bit  #weight resolution bits 
    A = Thr_bits   #accumulator threshold 
    #threshold = 1.0
    max_w = threshold/(2**A/2**(precision_bit-1))
    min_w = -max_w
    step = np.diff(np.linspace(min_w,max_w,2**precision_bit)[0:2])
    min_w = min_w#-step[0]
    max_w = max_w#+step[0]
        
    n = 2**precision_bit-1
    q = (max_w*2.0)/n
    q_list = np.round(np.arange(min_w,max_w,q),precision_bit) #quantize in range
    func = lambda x: q_list[np.abs(x-q_list).argmin()]
    q_list = np.array(list(map(func, x_list)))
    if len(x.shape) == 1:
        return q_list
    else:
        return q_list.reshape(x.shape)

flag = 1
def low_precision(state_dict,precision=8, threshold=1.0):
    """
    transfer the weight to low precision weight matrix
    :param state_dict: the .pth file of model
    :param precision: target precision like 8 bit
    :return: state_dict
    """
    for k in state_dict.keys():
        if k =='thr_h' or k == 'thr_h1' or k == 'thr_o' or k == 'thr_h2':
            #w = state_dict[k].data.cpu().numpy()
            n_thr = len(state_dict[k].data.cpu().numpy())
            #w = np.repeat(threshold, n_thr)
            w = state_dict[k].data.cpu().numpy()
        else:
            w = percision_transfer(state_dict[k].data.cpu().numpy(),precision_bit=precision, threshold=threshold)
            w[state_dict[k].data.cpu().numpy() == 0] = 0 # keep zero
        state_dict[k] = torch.tensor(w)
    return state_dict

#loads train and test images [0,1], flattened to (1, 28*28), and their labels
#data_set = np.load('dataset_12_class/all_data.npz', allow_pickle = True)['data']
data_set = np.empty(5500, dtype = object)
for i in range(int(5500/250)):
    chunk = np.load('dataset_12_class/datachunk_' + str(i) + '.npz', allow_pickle = True)['data']
    data_set[int(i*250):int((i+1)*250)] = chunk
    
labels_data = np.load('dataset_12_class/all_labels.npz', allow_pickle = True)['data']
 
T_sim = 28 #length of the time dimension, images will be transformed to (28*28, T_sim) joue

N_ex = data_set.shape[0]

#initialize TTFS-encoded images to zeros
data_set_in_time = np.zeros((data_set.shape[0], 1, 32, 32, T_sim))

#convert to TTFS arrays
for i in range(data_set.shape[0]):
    LL = data_set[i].shape[0]
    curr_port = data_set[i].reshape((LL, 32, 32))
    step_idx = int(np.floor(LL / T_sim))
    for j in range(T_sim):
        data_set_in_time[i,0,:,:,j] = (np.mean(curr_port[j*step_idx:(j+1)*step_idx,:,:], axis = 0) > 0).astype(int)
        #data_set_in_time[i,0,:,:,j] = (curr_port[j*step_idx,:,:] > 0).astype(int)
        #data_set_in_time[i,0,:,:,j] = (curr_port[j,:,:] > 0).astype(int)
    
    
hidden_dim = 240
output_dim = 12
batch_size = 128 #200

data_set_in_time, labels_data = shuffle(data_set_in_time, labels_data, random_state=0)

criterion = nn.NLLLoss()

conv_nbr_1 = 12 
conv_sz_1 = 5#7, 13
a_c_1_x = x_sz - conv_sz_1 + 1
a_c_1_y = y_sz - conv_sz_1 + 1

################ CHOOSE IF REUSE GOOD MODEL #################


Acc = []
conf_mat = []
b_e = 0

K = 6
hop = int(np.round(N_ex / K))
event_num = []
for k in range(K-1):
    model = torch.load("saved_models_12_class/eCNN4_" + str(k) + ".pt")
    
    test_set = data_set_in_time[k*hop:(k+1)*hop]
    test_label = labels_data[k*hop:(k+1)*hop]
    train_set = np.zeros((data_set_in_time.shape[0] - hop, data_set_in_time.shape[1], data_set_in_time.shape[2], data_set_in_time.shape[3], data_set_in_time.shape[4]))
    train_label = np.zeros(labels_data.shape[0]-hop)
    train_set[:k*hop] = data_set_in_time[:k*hop]
    train_label[:k*hop] = labels_data[:k*hop]
    train_set[(k)*hop:] = data_set_in_time[(k+1)*hop:]
    train_label[(k)*hop:] = labels_data[(k+1)*hop:]  
    
    low_precision_state = low_precision(model.state_dict(), precision=4)
    model.load_state_dict(low_precision_state)

    #Testing accuracy on test set
    c_spike =  torch.zeros(a_c_1_x*a_c_1_y) # random init -2 -2
    c_mem =  torch.zeros((conv_nbr_1, a_c_1_x, a_c_1_y)) # random init normal
    hidden_mem =  torch.zeros(model.hidden_size_1) # random init
    output_mem =  torch.zeros(model.output_size) #random init
    hidden_spike = torch.zeros(model.hidden_size_1)
    output_spike = torch.zeros(model.output_size)     
    
    print("Testing model")
    images = torch.from_numpy(test_set)
    labels = torch.from_numpy(test_label)
    pred_, l_, nbr_events_avrg = model.forward(images, labels, c_mem, hidden_mem,c_spike, hidden_spike, output_mem, output_spike)
    event_num.append(nbr_events_avrg.detach().numpy())
    test_accuracy = (pred_[0] == labels).sum().data.cpu().numpy() / float(len(pred_[0]))
    print('Test Acc: {:.4f}'.format(test_accuracy))
    
    print(test_accuracy)
    conf_mat.append(confusion_matrix(test_label, pred_[0]))
    Acc.append(test_accuracy)
    del model
    
    
print(conf_mat)
print(Acc)

print("Mean accuracy:" + str(np.mean(np.array(Acc))))
print("Standard" + str(np.std(np.array(Acc))))

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = conf_mat[0]
array2 = np.zeros(array.shape)
for i in range(array.shape[0]):
    for j in range(len(conf_mat)):
        array = conf_mat[j]
        array2[i,:] += array[i,:] / np.sum(array[i,:])

array2 = array2 / len(conf_mat)
array22 = array2[1:,1:]  

for i in range(array22.shape[0]):
    array22[i,:] = array22[i,:] / np.sum(array22[i,:])
    idx = np.argwhere(array22[i,:] <= 0.0025)[:,0]
    array22[i,idx] = 0

acc_acc = 0
for i in range(array22.shape[0]):
    acc_acc += array22[i,i]
acc_acc = acc_acc / array22.shape[0]
print(acc_acc) 
  
lli = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
df_cm = pd.DataFrame(array22, index = [i for i in lli],
                  columns = [i for i in lli])
plt.figure(figsize = (10,7))
ax = sn.heatmap(df_cm, annot=True, cbar=False, cmap = "YlGn")
ax.set(xlabel='Predicted label', ylabel='True label')

# =============================================================================
# plt.figure(1)
# event_num = np.array(event_num)
# mu_ev_num = np.mean(event_num, axis = 0)
# std_ev_num = np.std(event_num, axis = 0)
# plt.errorbar(np.linspace(0, 3, 4), mu_ev_num, yerr=std_ev_num, fmt='--.')
# maxx = np.max(event_num, axis = 0)
# plt.plot(np.linspace(0, 3, 4), maxx, '--.')
# plt.xticks(np.linspace(0, 3, 4))
# plt.grid('on')
# plt.legend(["max","mean and std dev"])
# plt.xlabel('Time step')
# plt.ylabel('Average number of events')
# =============================================================================


