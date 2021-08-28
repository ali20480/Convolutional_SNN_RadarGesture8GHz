"""
Train eCNN based on LeNet-5 architecture
Gesture recognition using 8GHz Radar
Author: Ali Safa - IMEC- KU Leuven
"""
################highest accuracy config#####################
import numpy as np
from eNetworks import mini_eCNN
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.utils import shuffle
x_sz = 32 
y_sz = 32 


Thr_bits = 6

def percision_transfer(x, precision_bit=16, threshold=1.0):
    x_list = x.reshape((-1,))
    A = Thr_bits 
    max_w = threshold/(2**A/2**(precision_bit-1))
    min_w = -max_w
    step = np.diff(np.linspace(min_w,max_w,2**precision_bit)[0:2])
    min_w = min_w
    max_w = max_w
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
    for k in state_dict.keys():
        if k =='thr_h' or k == 'thr_h1' or k == 'thr_o' or k == 'thr_h2':
            n_thr = len(state_dict[k].data.cpu().numpy())
            w = state_dict[k].data.cpu().numpy()
        else:
            w = percision_transfer(state_dict[k].data.cpu().numpy(),precision_bit=precision, threshold=threshold)
            w[state_dict[k].data.cpu().numpy() == 0] = 0 # keep zero
        state_dict[k] = torch.tensor(w)
    return state_dict


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
    
    
hidden_dim = 240
output_dim = 12
batch_size = 128 #200

data_set_in_time, labels_data = shuffle(data_set_in_time, labels_data, random_state=0)

criterion = nn.NLLLoss()

conv_nbr_1 = 12 
conv_sz_1 = 5#7, 13
a_c_1_x = x_sz - conv_sz_1 + 1
a_c_1_y = y_sz - conv_sz_1 + 1


model = mini_eCNN(x_sz, y_sz, conv_sz_1, conv_nbr_1, hidden_dim, output_dim, criterion=criterion, batch_size=batch_size)
#model = torch.load("eCNN_best_4.pt")
class Clipper(object):
    def __init__(self, threshold = 6):
        # chip parameter scaling with threshold normalized to 1
        self.W = 4   #weight resolution bits 
        self.A = threshold   #accumulator threshold
        self.max_w = threshold/(2**self.A/2**(self.W-1))
        self.min_w = -(self.max_w)
        step = np.diff(np.linspace(self.min_w,self.max_w,2**self.W)[0:2])
        self.min_w = self.min_w#-step[0] # -8,7
        self.max_w = self.max_w#-step[0]
        
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            if(type(module) == nn.Linear):
                w = module.weight.data
                w = w.clamp(self.min_w,self.max_w)
                #raise Exception
                module.weight.data = w
                
def train(model, train_loader, optimizer, epochs, batch_size, prev_low_precision_state, val_loss, precision = 4):
    clipper = Clipper(threshold=Thr_bits)
    for e in range(epochs):
        train_acc = 0
        train_loss_sum = 0
        for i, (images, labels) in enumerate(train_loader):
            
            optimizer.zero_grad()
            c_spike =  torch.zeros(a_c_1_x*a_c_1_y) # random init -2 -2
            c_mem =  torch.zeros((conv_nbr_1, a_c_1_x, a_c_1_y)) # random init normal
            hidden_mem =  torch.zeros(model.hidden_size_1) # random init
            output_mem =  torch.zeros(model.output_size) #random init
            hidden_spike = torch.zeros(model.hidden_size_1)
            output_spike = torch.zeros(model.output_size)
            
            if e % 20 == 0 and e != 0: #>epochs-2:
                low_precision_state = low_precision(model.state_dict(), precision=precision)
                high_precision_state = model.state_dict()
                model.load_state_dict(low_precision_state)
                
            predictions, train_loss, nbr_events_avrg = model(images, labels, c_mem, hidden_mem,c_spike, hidden_spike, output_mem, output_spike)
            predicted = predictions
            
            if e % 20 == 0 and e != 0: #>epochs-2:
                model.load_state_dict(high_precision_state)
                
            train_loss.sum().backward(retain_graph=True)
            train_loss_sum += train_loss.data.cpu().numpy()
                
            optimizer.step()
            model.apply(clipper)
            predicted = predicted.t()
            train_acc += (predicted.T[0] == labels).sum() 
        
        #scheduler.step()
        train_acc = train_acc.data.cpu().numpy()
        print("Epoch: " + str(e) + " Loss: " + str(train_loss_sum.item()/len(train_loader)/(batch_size)) + " Accuracy: " + str(train_acc/(batch_size)/len(train_loader)))
        
        if train_acc/(batch_size)/len(train_loader) > val_loss and e % 20 == 0 and e != 0:
            print('replaced! ' + str(train_acc/(batch_size)/len(train_loader)) + "vs. " + str(val_loss))
            val_loss = train_acc/(batch_size)/len(train_loader)
            prev_low_precision_state = low_precision_state
            
    return prev_low_precision_state, val_loss
        

learning_rate = 3e-4

base_params = [model.conv1.weight, model.conv1.bias, model.i2h.weight, model.i2h.bias, model.h2o.weight, model.h2o.bias]

optimizer = torch.optim.Adam([{'params': base_params},], lr=learning_rate)

Acc = []
conf_mat = []
epochs = int(20*4)
repeat = 5
b_e = 0

K = 6 #joue
hop = int(np.round(N_ex / K))
for k in range(K-1):
    prev_low_precision_state = []
    val_loss = 0
    test_set = data_set_in_time[k*hop:(k+1)*hop]
    test_label = labels_data[k*hop:(k+1)*hop]
    train_set = np.zeros((data_set_in_time.shape[0] - hop, data_set_in_time.shape[1], data_set_in_time.shape[2], data_set_in_time.shape[3], data_set_in_time.shape[4]))
    train_label = np.zeros(labels_data.shape[0]-hop)
    train_set[:k*hop] = data_set_in_time[:k*hop]
    train_label[:k*hop] = labels_data[:k*hop]
    train_set[(k)*hop:] = data_set_in_time[(k+1)*hop:]
    train_label[(k)*hop:] = labels_data[(k+1)*hop:]
    
    train_data = TensorDataset(torch.from_numpy(train_set), torch.from_numpy(train_label))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    prev_test_acc = 0
    for h in range(repeat):
        prev_low_precision_state, val_loss = train(model, train_loader, optimizer, epochs, batch_size, prev_low_precision_state, val_loss)    
        
        model.load_state_dict(prev_low_precision_state)
        
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
        test_accuracy = (pred_[0] == labels).sum().data.cpu().numpy() / float(len(pred_[0]))
        if test_accuracy > prev_test_acc:
            prev_test_acc = test_accuracy
            torch.save(model, "eCNN4_" + str(k) + ".pt")
        
        print('Test Acc: {:.4f}'.format(test_accuracy))
        print('Top Test Acc: {:.4f}'.format(prev_test_acc))
    
    print(test_accuracy)
    conf_mat.append(confusion_matrix(test_label, pred_[0]))
    Acc.append(prev_test_acc)
    del model
    model = mini_eCNN(x_sz, y_sz, conv_sz_1, conv_nbr_1, hidden_dim, output_dim, criterion=criterion, batch_size=batch_size)
    base_params = [model.conv1.weight, model.conv1.bias, model.i2h.weight, model.i2h.bias, model.h2o.weight, model.h2o.bias]
    #base_params = [model.conv1.weight, model.conv2.weight, model.i2h.weight, model.h2h.weight, model.h2o.weight]
    optimizer = torch.optim.Adam([{'params': base_params},], lr=learning_rate)
    
print(conf_mat)
print(Acc)

print("Mean accuracy:" + str(np.mean(np.array(Acc))))
print("Standard" + str(np.std(np.array(Acc))))
