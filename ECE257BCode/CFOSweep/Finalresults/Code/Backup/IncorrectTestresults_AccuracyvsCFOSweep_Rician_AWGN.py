
import numpy as np
import pickle
import datetime
from ModulationPredictionResNet import ModulationPredictionResNet
from ModulationPredictionCNN import ModulationPredictionCNN
import torch
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import math
from datetime import datetime
import pickle
SNRLevels = list(range(20,-21,-2))
modulationTypes = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", "BFM", "DSBAM", "SSBAM"]
label_dict = {'16QAM':0, '64QAM':1, '8PSK':2, 'BFM':3, 'BPSK':4, 'CPFSK':5, 'DSBAM':6, 'GFSK':7,
     'PAM4':8, 'QPSK':9, 'SSBAM':10}



def Testdata(data, label, model_file, model_type):
    shuffleidx = np.arange(data.shape[0])
    np.random.shuffle(shuffleidx)
    x_test = data[shuffleidx]
    y_test = label[shuffleidx]

    def calculate_accuracy(model, data, label, batch_size, computing_device):
        n_samples = data.shape[0]
        n_minibatch = int((n_samples+batch_size-1)/batch_size)
        accuracy = 0
        I = np.arange(n_samples)
        predictLabelAll = np.empty((0,))
        trueLabelAll = np.empty((0,))
        accuracyperMod = np.zeros((len(modulationTypes), ))
        for i in range(n_minibatch):
            idx = I[batch_size*i:min(batch_size*(i+1), n_samples)]
            dt = data[idx].to(computing_device)
            lbl = label[idx]
            output = model(dt).detach()
            output = output.cpu().numpy()
            output = np.argmax(output,axis=1)
            predictLabelAll = np.append(predictLabelAll, output)
            trueLabelAll = np.append(trueLabelAll, lbl)
            accuracy += np.sum(output == lbl)
        
        idx2 = 0

        return predictLabelAll, trueLabelAll
    if model_type == 'Resnet':
        model = ModulationPredictionResNet()
    else:
        model = ModulationPredictionCNN('relu')
            
    batch_size = 1000
    computing_device = torch.device("cuda")
    #print('Model file used: ', model_file)
    model.load_state_dict(torch.load(model_file))
    model.to(computing_device)

    x_test = torch.tensor(x_test).float().to(computing_device)
    PredictedLabel, trueLabelAll = calculate_accuracy(model, x_test, y_test, batch_size, computing_device)
    #print('Test Accuracy for SNR: ', SNRvalue, ' is = ', accuracy)
    return PredictedLabel, trueLabelAll # this returns predicted label for all data belonging to single SNR


#data = databysnrbymod
modulationTypes = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", "BFM", "DSBAM", "SSBAM"]

fs_range = [200*pow(10,3), 1.25*pow(10,6)]
no_cfo_steps = 400
step1 = int(6e3/no_cfo_steps)
step2 = int(400/no_cfo_steps)


CFO_ranges = [range(-3000, 3000, step1), range(-200,201,step2)]

modelfilelocation = '/home/vesathya/ModulationClassification/ECE257B/Models/CFOSweep/'
model_file1 = modelfilelocation + 'CNN_Alldata.pt'
model_file2 = modelfilelocation + 'Resnet_Alldata.pt'
model_file3 = modelfilelocation + 'Resnet_model_ece228.pt'
model_file4 = modelfilelocation + 'CNN_model_ece228.pt'

model_file_list = [model_file1, model_file2, model_file3, model_file4]

snr_range = list(range(-20,21,2))
acc_dict = {}
testdatatypes = ['Rician', 'AWGN']
num_frames = data1.shape[3]

# generating labels - same for all data of a particular datatype,SNR 
label = np.ones((num_frames*len(modulationTypes), ))
modidx = 0
for modval in modulationTypes:
    label[modidx*num_frames:(modidx+1)*num_frames] = label_dict[modval]*np.ones((num_frames, ))
    
for modelfileidx in range(len(model_file_list)):
    if 'Alldata' in model_file_list[modelfileidx]:
        fs = fs_range[1]
        data = data1 #TrueSweepRicianAWGN_CFO_0_SNR_minus20to20_nf_100_fs1pt25e6.mat
        CFO_range = CFO_ranges[1]
    else:
        fs = fs_range[0]
        data = data0 #TrueSweepRicianAWGN_CFO_0_SNR_minus20to20_nf_100_fs200e3.mat
        CFO_range = CFO_ranges[0]
    if 'Resnet' in model_file_list[modelfileidx]:
        model_type = 'Resnet'
    else:
        model_type = 'CNN'   
    for testdataidx in range(len(testdatatypes)):
        print(model_file_list[modelfileidx], testdatatypes[testdataidx])
        for snridx in range(len(snr_range)):
            data_temp1 = data[testdataidx,snridx,:,:,:,:]
            data_temp2 = data_temp1.reshape((-1,2,1024))
            data_temp2i = data_temp2[:,0,:]
            data_temp2q = data_temp2[:,1,:]
            data_temp3 = np.empty(data_temp2.shape)
            for CFOidx in range(len(CFO_range)):
                cos_array = np.cos(2*math.pi*CFO_range[CFOidx]*np.array(range(1024))*(1/fs))
                sin_array = np.sin(2*math.pi*CFO_range[CFOidx]*np.array(range(1024))*(1/fs))
                
                data_temp2i_corrected = np.multiply(data_temp2i, cos_array)
                data_temp2q_corrected = np.multiply(data_temp2i, sin_array)
#                 data_temp3 = np.concatenate((data_temp2i_corrected, data_temp2q_corrected), axis = 1)
                data_temp3[:,0,:] = data_temp2i_corrected
                data_temp3[:,1,:] = data_temp2q_corrected
                PredictedLabel, trueLabelAll = Testdata(data_temp3, label, model_file_list[modelfileidx]\
                                                       , model_type)
                acc = np.sum(PredictedLabel == trueLabelAll)/trueLabelAll.shape[0]
                #print(model_file_list[modelfileidx], testdatatypes[testdataidx], snr_range[snridx],\
                #      CFO_range[CFOidx],acc)
                acc_dict[model_file_list[modelfileidx], testdatatypes[testdataidx], CFO_range[CFOidx],\
                         snr_range[snridx]] = acc
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
filelocation = '/home/vesathya/ModulationClassification/ECE257B/Models/CFOSweep/'
accuracydictfilename = filelocation + 'Accuracydict_trueCFOSweep' + dt_string + '.pckl'
f = open(accuracydictfilename, 'wb')
pickle.dump(acc_dict, f)
f.close()
