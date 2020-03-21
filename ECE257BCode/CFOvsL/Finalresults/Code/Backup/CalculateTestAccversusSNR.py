
from ModulationPredictionCNNL128 import ModulationPredictionCNNL128
from ModulationPredictionCNNL256 import ModulationPredictionCNNL256
from ModulationPredictionCNNL512 import ModulationPredictionCNNL512
from ModulationPredictionCNNL768 import ModulationPredictionCNNL768
from ModulationPredictionCNNL1024 import ModulationPredictionCNNL1024
from ModulationPredictionCNNLbeyond768 import ModulationPredictionCNNLbeyond768
import torch.nn as nn
import torch
from Train import train_model
import sys
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torch
import pickle

from datetime import datetime

## 1. Num epochs changed to 3 for dry run. Change it back for actual testing ot 100
label_dict = {'QAM16':0, 'QAM64':1, '8PSK':2, 'WBFM':3, 'BPSK':4, 'CPFSK':5, 'AM-DSB':6, 'GFSK':7,
     'PAM4':8, 'QPSK':9, 'AM-SSB':10}

def testdata(modelclass, model_file, data, label,L):
    activation = 'relu'
    x_data, x_test, y_data, y_test = train_test_split(data, label, test_size=0.98, random_state=1)
    
    def calculate_accuracy(model, data, label, batch_size, computing_device):
        n_samples = data.shape[0]
        n_minibatch = int((n_samples+batch_size-1)/batch_size)
        print("minibatch value is ", n_minibatch)
        accuracy = 0
        I = np.arange(n_samples)
        for i in range(n_minibatch):
            idx = I[batch_size*i:min(batch_size*(i+1), n_samples)]
            dt = data[idx].to(computing_device)
            #print(idx, label)
            lbl = label[idx]
            output = model(dt).detach()
            output = output.cpu().numpy()
            output = np.argmax(output,axis=1)
            accuracy += np.sum(output == lbl)
        return accuracy/n_samples
    
    if L < 768:
        model = modelclass(activation)
    else:
        #print(L)
        model = modelclass(activation, int(L/32))
        #print(model)
#     model = modelclass(activation='relu')
    batch_size = 1000
    computing_device = torch.device("cuda")
    model.load_state_dict(torch.load(model_file))
    model.to(computing_device)

    x_test = torch.tensor(x_test).float().to(computing_device)
    accuracytest = calculate_accuracy(model, x_test, y_test, batch_size, computing_device)
    print('Test Accuracy = ', accuracytest)

    return accuracytest

#L_range = [128, 256, 512, 768, 1024]
#modelrange = [ModulationPredictionCNNL128, ModulationPredictionCNNL256, ModulationPredictionCNNL512, \
#        ModulationPredictionCNNL768, ModulationPredictionCNNL1024]
L_range = [128, 256, 512, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408]
# L_range = [128, 256, 512, 768, 832, 896, 960, 1024, 1088, 1152]

modelrange = [ModulationPredictionCNNL128, ModulationPredictionCNNL256, ModulationPredictionCNNL512]

# Note that for any values equal and beyond 768, we have 6 conv layers and we call teh same class with 
# an addiiotnal input of the kernel size of he final average pooling layer.
for i in range(3,len(L_range)):
    modelrange.append(ModulationPredictionCNNLbeyond768)
    
CFOmaxdev_range = [0,500]
modulationTypes = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", "BFM", "DSBAM", "SSBAM"]
acc_dict = {}

# modelfile_location = '/home/vesathya/ModulationClassification/ECE257B/Models/CFOvsL/Run1_128to1408result_slowerlearningrate/'
modelfile_location = '/home/vesathya/ModulationClassification/ECE257B/Models/CFOvsL/Finalresults/Models/Slowrateuntil1408_CFO_0_500/'
for CFOmaxdev in CFOmaxdev_range:
    Lidx = 0
#     filename = '/home/vesathya/ModulationClassification/ECE257B/Data/CFOvsL/CFO_'+\
#     str(CFOmaxdev) +'_L1024.pckl'
    
    filename = '/home/vesathya/ModulationClassification/ECE257B/Data/CFOvsL/'+\
               'RML2016.10a_dict_cfo_' + str(CFOmaxdev) +'_framesize_1800.dat'
    
    #ModulationClassification ECE257B Data CFOvsL
    print("Dataset Filename is:  ", filename)
    f = open(filename,'rb')
    dataset = pickle.load(f,encoding='latin1')
    f.close()
    snrs,mods = map(lambda j: sorted(list  (set  (map(lambda x: x[j], dataset.keys())  )  )), [1,0])
#     X = []  
#     lbl = []
    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(dataset[(mod,snr)])
            for i in range(dataset[(mod,snr)].shape[0]):  lbl.append((mod,snr))
                
#Run1_128to1408result_CFOvsLmaxdev_0_500_L128to1408_100Epochs_08_03_2020_12_16_41
# CNNoverRadioML_CFOmaxdev_0_L_1024_100EpochsRun1_model.pt
    for L in L_range:
#         model_file = modelfile_location + 'CNNoverRadioML_CFOmaxdev_' + str(CFOmaxdev) \
#                      + '_L_' + str(L) + '_100EpochsRun1_model.pt'

        model_file = modelfile_location + 'CNNoverRadioML_CFOmaxdev_' + str(CFOmaxdev)+ '_L_'+str(L) + \
                        '__slowlearn_100EpochsRun1_model.pt'
        modelclass  = modelrange[Lidx]
        for snr in snrs:
            lbl = []
            tempdata = []
            for mod in mods:
                tempdata.append(dataset[(mod,snr)])
                for i in range(dataset[(mod,snr)].shape[0]):  lbl.append((mod,snr))
            tempdata = np.vstack(tempdata)
            label_val = list(map(lambda x: lbl[x][0],range(len(lbl))))
            label_numbers = list(map(lambda x: label_dict[x], label_val)) #label_dict[label_val]
            label_numbers = np.array(label_numbers)
                
            data = tempdata[:,:,0:L]
            label = label_numbers
            print('SNR is: ', snr)
            acc_dict[CFOmaxdev, L, snr] = testdata(modelclass, model_file, data, label,L)

#                 data = dataset[(mod,snr)][:,:,0:L]
#                 label = np.ones((dataset[(mod,snr)].shape[0],))*label_dict[mod]
#                 print('Mod type is: ',mod, 'SNR is: ', snr)
#                 acc_dict[CFOmaxdev, L, snr, mod] = testdata(modelclass, model_file, data, label)
                
        Lidx = Lidx + 1
#                 X.append(dataset[(mod,snr)])
#                 for i in range(dataset[(mod,snr)].shape[0]):  lbl.append((mod,snr))
#     X = np.vstack(X)
#     label_val = list(map(lambda x: lbl[x][0],range(len(lbl))))
#     label_numbers = list(map(lambda x: label_dict[x], label_val)) #label_dict[label_val]
#     label_numbers = np.array(label_numbers)