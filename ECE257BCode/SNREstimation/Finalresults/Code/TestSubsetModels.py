


## Subset ranges in SNR are: [0,2; 4,6,8; 10,12,14; 16,18,20] - lowest subset has one region lesser - 
## harder to learn for lower SNR.

### I have swapped the SNR regions of subset 1 and 4 to verify something. CHange it back. 
### Values swapped back - it was used tofind a bug where the range which is origianlly 
### from 20 to -20 was though of as gfoing from -20 to 20


# Dictionary of SNR subsets
databySNRsubsetdict = {}

SNRregions = {'SNRsubset1': [0,2], 'SNRsubset2': [4,6,8],'SNRsubset3': [10,12,14],\
              'SNRsubset4': [16,18,20], 'SNRsubset5':list(range(-20,0,2)) }

for snrsubset in SNRregions.keys():
    
    # 0 SNR corresponds to index of 10(11th value). Therefore (-SNR)/2 + 10 is the correct mapping of SNR to
    # indexing within the context of 20:-20:-2, where 20 corresponds to 0 and -20 to 20 and 0 to 10.
    
    num_snr_subset = len(SNRregions[snrsubset])
    temp2 = data[(-1*np.array(SNRregions[snrsubset])/2 +10).astype(int)]
    databySNRsubsetdict[snrsubset + '_data'] = temp2.reshape((-1,2,L))
    databySNRsubsetdict[snrsubset + '_label'] = np.array(list(label[:])*num_snr_subset)
       
## Generate l=128 length data to be fed to this.. run trim to L=128 before running this.

from ModulationPredictionCNNL128 import ModulationPredictionCNNL128
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

def calculate_accuracy(model, data, label, batch_size, computing_device):
    n_samples = data.shape[0]
    n_minibatch = int((n_samples+batch_size-1)/batch_size)
    accuracy = 0
    I = np.arange(n_samples)
    for i in range(n_minibatch):
        idx = I[batch_size*i:min(batch_size*(i+1), n_samples)]
        dt = data[idx].to(computing_device)
        lbl = label[idx]
        output = model(dt).detach()
        output = output.cpu().numpy()
        output = np.argmax(output,axis=1)

        accuracy += np.sum(output == lbl)

    return accuracy/n_samples
acc_dict_snrall = {}
snrsubsetidx = 1
for snrsubset in SNRregions.keys():
    
    model_file = '/home/vesathya/ModulationClassification/ECE257B/Models/SNREstimation/' + \
                    'CNN_OTARayleigh_L128_100epochsRun1_SNRsubset' + str(snrsubsetidx) + '_model.pt'
    print('Data shape is: ', databySNRsubsetdict[snrsubset + '_data'].shape, \
          ' and Label shape is ',databySNRsubsetdict[snrsubset + '_label'].shape)
    activation = 'relu'
    model = ModulationPredictionCNNL128(activation=activation)
    batch_size = 1000
    computing_device = torch.device("cuda")
    model.load_state_dict(torch.load(model_file))
    model.to(computing_device)

    x_data, x_test, y_data, y_test = train_test_split(databySNRsubsetdict[snrsubset + '_data'],\
                                      databySNRsubsetdict[snrsubset + '_label'], test_size=0.98, random_state=1)

    x_test = torch.tensor(x_test).float().to(computing_device)
    accuracy = calculate_accuracy(model, x_test, y_test, batch_size, computing_device)
    print('Test Accuracy for SNR values ', SNRregions[snrsubset], " is: ", accuracy)
    acc_dict_snrall[snrsubset] = [SNRregions[snrsubset], accuracy] 
    snrsubsetidx = snrsubsetidx + 1
