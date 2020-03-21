

from ModulationPredictionCNNL128 import ModulationPredictionCNNL128
import torch.nn as nn
import torch
from Train import train_model
import sys
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torch



## We have commentd out the code of label2onehot and onehottolabel. 
## If the program gives error, uncomment that and explore on that front.

## We have changed the no. of epochs to 3 just for a dry run. Change it back to 100.
## Changed the num_epochs back to 100.


databySNRsubsetdict = {}

SNRregions = {'SNRsubset1': [0,2], 'SNRsubset2': [4,6,8],'SNRsubset3': [10,12,14],'SNRsubset4': [16,18,20] }

for snrsubset in SNRregions.keys():
    
    # 0 SNR corresponds to index of 10(11th value). Therefore (-SNR)/2 + 10 is the correct mapping of SNR to
    # indexing within the context of 20:-20:-2, where 20 corresponds to 0 and -20 to 20 and 0 to 10.
    
    num_snr_subset = len(SNRregions[snrsubset])
    temp2 = data[(-1*np.array(SNRregions[snrsubset])/2 +10).astype(int)]
    databySNRsubsetdict[snrsubset + '_data'] = temp2.reshape((-1,2,L))
    databySNRsubsetdict[snrsubset + '_label'] = np.array(list(label[:])*num_snr_subset)
    
    
def loadSplitTrain(Savefilename, data, label):

    # Split test data
    activation = 'relu'
    model = ModulationPredictionCNNL128(activation=activation)
    x_data, x_test, y_data, y_test = train_test_split(data, label, test_size=0.2, random_state=1)
    # Split validation data
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=1)


    train_set = {'data':torch.tensor(x_train).float(), 'labels':torch.tensor(y_train).float() }
    val_set = {'data':torch.tensor(x_val).float(), 'labels':torch.tensor(y_val).float()}
    print(x_train.shape, x_val.shape)
    del data


    # ## Train Model

    model_file = 'Models/'+ Savefilename + '_model.pt'
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.00001
    criterion = nn.CrossEntropyLoss()
    computing_device = torch.device("cuda")
    print('Model used: ', model_file)
    model1, Loss, Accuracy = train_model(model, model_file, train_set, val_set, num_epochs, batch_size, \
                                        learning_rate, criterion, computing_device)
    
    def calculate_accuracy(model, data, label, batch_size, computing_device):
        n_samples = data.shape[0]
        n_minibatch = int((n_samples+batch_size-1)/batch_size)
        print("minibatch value is ", n_minibatch)
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

    model = ModulationPredictionCNN(activation)
    activation = 'relu'
    model = ModulationPredictionCNNL128(activation=activation)
    batch_size = 1000
    computing_device = torch.device("cuda")
    model.load_state_dict(torch.load(model_file))
    model.to(computing_device)

    x_test = torch.tensor(x_test).float().to(computing_device)
    accuracy = calculate_accuracy(model, x_test, y_test, batch_size, computing_device)
    print('Test Accuracy = ', accuracy)

    x_train = torch.tensor(x_train).float().to(computing_device)
    accuracy = calculate_accuracy(model, x_train, y_train, batch_size, computing_device)
    print('Train Accuracy = ', accuracy)
    
    import matplotlib.pyplot as plt
    %matplotlib inline
    epochs = [i for i in range(len(Loss['train']))]
    plt.plot(epochs, Loss['train'])
    plt.plot(epochs, Loss['valid'])
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.savefig(activation+'_model_loss')
    plt.show()

    plt.plot(epochs, Accuracy['train'])
    plt.plot(epochs, Accuracy['valid'])
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])
    plt.savefig(activation+'_model_acc')
    plt.show()

for snrsubset in SNRregions.keys():
    
    
    Savefilename = 'CNN_OTARayleigh_L128_100epochsRun1' + '_' + snrsubset
    print(databySNRsubsetdict[snrsubset + '_data'].shape)
    loadSplitTrain(Savefilename,databySNRsubsetdict[snrsubset + '_data'],\
                   databySNRsubsetdict[snrsubset + '_label'])
