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
x_sz = 100 #100 #100
y_sz = 48 #48 50

def percision_transfer(x, precision_bit=16, result_p = 4):

    if len(x.shape) == 1:
        o = x.shape
    elif len(x.shape) == 2:
        i,o = x.shape
    elif len(x.shape) == 4:
        i,o,v,c = x.shape
    x_list = x.reshape((-1,))
    n = 2**precision_bit-1
    q = 1./n
    q_list = np.round(np.arange(-1,1+q,q),result_p)
    func = lambda x: q_list[np.abs(x-q_list).argmin()]
    q_list = np.array(list(map(func, x_list)))
    if len(x.shape) == 1:
        return q_list
    elif len(x.shape) == 2:
        return q_list.reshape((i,o))
    elif len(x.shape) == 4:
        return q_list.reshape((i,o,v,c))

flag = 1
def low_precision(state_dict,precision=4):
    """
    transfer the weight to low precision weight matrix
    :param state_dict: the .pth file of model
    :param precision: target precision like 8 bit
    :return: state_dict
    """
    for k in state_dict.keys():
        if k =='thr_h':
            w = state_dict[k].data.cpu().numpy()
        else:
            w = percision_transfer(state_dict[k].data.cpu().numpy(),precision_bit=precision)
        state_dict[k] = torch.tensor(w)
    return state_dict

#loads train and test images [0,1], flattened to (1, 28*28), and their labels
with open('pre_processed_dataset_5_class/data_set_k48_B50_z2.npy', 'rb') as f:
    data_set = np.load(f)
    #Test_set_down = np.reshape(Test_set_down, (Test_set_down.shape[0], Test_set_down.shape[1], x_sz, y_sz))
    
with open('pre_processed_dataset_5_class/labels_k48_B50_z2.npy', 'rb') as f:
    labels_data = np.load(f)
    
    
T_sim = 4 #length of the time dimension, images will be transformed to (28*28, T_sim)

N_ex = data_set.shape[0]

#initialize TTFS-encoded images to zeros
data_set_in_time = np.zeros((data_set.shape[0], data_set.shape[1], data_set.shape[2],data_set.shape[3], T_sim+1))

#convert to TTFS arrays
for i in range(data_set.shape[0]):
    curr_port = data_set[i,:,:,:]
    t_e = np.round((1 - curr_port) * (T_sim))
    for j in range(t_e.shape[1]):
        for k in range(t_e.shape[2]):
            if t_e[0,j,k] != t_e[0,j,k]:
                t_e[0,j,k] = 0
            data_set_in_time[i,:,j,k,int(t_e[0,j,k])] = 1
            #data_set_in_time[i,:,j,k,::int(t_e[0,j,k]+1)] = 1
            
data_set_in_time =  data_set_in_time[:,:,:,:,1:] #reject last time bin -> should not spike if zero!    
    
input_dim_rnn = data_set.shape[2]
hidden_dim = 120
output_dim = 5
batch_size = 128 #200

data_set_in_time, labels_data = shuffle(data_set_in_time, labels_data, random_state=0)

criterion = nn.NLLLoss()

conv_nbr_1 = 6 
conv_sz_1 = 13#7, 13
a_c_1_x = x_sz - conv_sz_1 + 1
a_c_1_y = y_sz - conv_sz_1 + 1

################ CHOOSE IF REUSE GOOD MODEL #################


Acc = []
conf_mat = []
b_e = 0

K = 6
hop = int(np.round(N_ex / K))
event_num = []
for k in range(K):
    model = torch.load("saved_models_5_class/mini_eCNN_" + str(k) + ".pt")
    
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



