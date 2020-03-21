
from ModulationPredictionCNNL128 import ModulationPredictionCNNL128
from ModulationPredictionCNNL256 import ModulationPredictionCNNL256
from ModulationPredictionCNNL512 import ModulationPredictionCNNL512
from ModulationPredictionCNNL768 import ModulationPredictionCNNL768
from ModulationPredictionCNNL1024 import ModulationPredictionCNNL1024
from ModulationPredictionCNNLbeyond768 import ModulationPredictionCNNLbeyond768
import torch.nn as nn
import torch
from Train_stopconditionchange_num5 import train_model
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

def loadSplitTrain(modelclass, Savefilename, data, label,L):
    activation='relu'

    # Split test data
#     activation = 'relu'
#     model = ModulationPredictionCNNL128(activation=activation)
    if L < 768:
        model = modelclass(activation)
    else:
        model = modelclass(activation, int(L/32))
            
    x_data, x_test, y_data, y_test = train_test_split(data, label, test_size=0.2, random_state=1)
    # Split validation data
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=1)


    train_set = {'data':torch.tensor(x_train).float(), 'labels':torch.tensor(y_train).float() }
    val_set = {'data':torch.tensor(x_val).float(), 'labels':torch.tensor(y_val).float()}
    print(x_train.shape, x_val.shape)
    #del data


    # ## Train Model

    model_file = 'Models/'+ Savefilename + '_model.pt'
    num_epochs = 200
    batch_size = 32
    learning_rate = 0.000001
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
            #print(idx, label)
            lbl = label[idx]
            output = model(dt).detach()
            output = output.cpu().numpy()
            output = np.argmax(output,axis=1)

            accuracy += np.sum(output == lbl)

        return accuracy/n_samples

    #model = ModulationPredictionCNN(activation)
#     model = modelclass(activation='relu')
    
    if L < 768:
        model = modelclass(activation)
    else:
        model = modelclass(activation, int(L/32))
        
        
    batch_size = 1000
    computing_device = torch.device("cuda")
    model.load_state_dict(torch.load(model_file))
    model.to(computing_device)

    x_test = torch.tensor(x_test).float().to(computing_device)
    accuracytest = calculate_accuracy(model, x_test, y_test, batch_size, computing_device)
    print('Test Accuracy = ', accuracytest)

    x_train = torch.tensor(x_train).float().to(computing_device)
    accuracytrain = calculate_accuracy(model, x_train, y_train, batch_size, computing_device)
    print('Train Accuracy = ', accuracytrain)
    
    import matplotlib.pyplot as plt
    #%matplotlib inline
    epochs = [i for i in range(len(Loss['train']))]
    plt.plot(epochs, Loss['train'])
    plt.plot(epochs, Loss['valid'])
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.savefig('model_loss')
    plt.show()

    plt.plot(epochs, Accuracy['train'])
    plt.plot(epochs, Accuracy['valid'])
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])
    plt.savefig('model_acc')
    plt.show()
    return accuracytest

#!cat >> /home/vesathya/ModulationClassification/ECE257B/Logs/Logs1.txt
L_range = [128, 256, 512, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408]
modelrange = [ModulationPredictionCNNL128, ModulationPredictionCNNL256, ModulationPredictionCNNL512]

# Note that for any values equal and beyond 768, we have 6 conv layers and we call teh same class with 
# an addiiotnal input of the kernel size of he final average pooling layer.
for i in range(3,len(L_range)):
    modelrange.append(ModulationPredictionCNNLbeyond768)

CFOmaxdev_range = [0, 500]

acc_dict = {}
for CFOmaxdev in CFOmaxdev_range:
    #filename = '/home/vesathya/ModulationClassification/ECE257B/Data/CFOvsL/CFO_'+\
    #str(CFOmaxdev) +'_L1024.pckl'
    filename = '/home/vesathya/ModulationClassification/ECE257B/Data/CFOvsL/'+\
               'RML2016.10a_dict_cfo_' + str(CFOmaxdev) +'_framesize_1800.dat'
    
    #ModulationClassification ECE257B Data CFOvsL
    print("Dataset Filename is:  ", filename)
    f = open(filename,'rb')
    dataset = pickle.load(f,encoding='latin1')
    f.close()
    snrs,mods = map(lambda j: sorted(list  (set  (map(lambda x: x[j], dataset.keys())  )  )), [1,0])
    X = []  
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(dataset[(mod,snr)])
            for i in range(dataset[(mod,snr)].shape[0]):  lbl.append((mod,snr))
    X = np.vstack(X)
    label_val = list(map(lambda x: lbl[x][0],range(len(lbl))))
    label_numbers = list(map(lambda x: label_dict[x], label_val)) #label_dict[label_val]
    label_numbers = np.array(label_numbers)
    #label_numbers = list(map(lambda x: lbl[x][0],range(len(lbl))))
    print("Data shape and label shape before truncation: ",X.shape, len(label_numbers))
    for i in range(len(L_range)):
        #L = 128
        data = X[:,:,0:L_range[i]]
        print("Data shape and label shape after truncation: ",data.shape, len(label_numbers))

        #data = dataRayleighL128
        #label = labelRayleigh
        #activation = 'relu'
        #model = ModulationPredictionCNNL128(activation=activation)
        modelclass  = modelrange[i]
        Modelfilename = 'CNNoverRadioML_CFOmaxdev_' + str(CFOmaxdev)+ '_L_'+str(L_range[i]) + \
                        '__slowlearn_100EpochsRun1'
        #Savefilename = 'CNN_OTARayleigh_L128_100epochsRun1' + '_' + snrsubset
        #print(databySNRsubsetdict[snrsubset + '_data'].shape)
        testacc = loadSplitTrain(modelclass, Modelfilename,data,label_numbers,L_range[i])
        print(testacc)
        acc_dict[(CFOmaxdev,L_range[i])] = testacc

now1 = datetime.now()
dt_string = now1.strftime("%d_%m_%Y_%H_%M_%S")
filename = '/home/vesathya/ModulationClassification/ECE257B/Models/CFOvsL/' + \
            'CFOvsLmaxdev_0_500_L128to1408_slowlearnRun2' + dt_string + '.pckl'
outfile = open(filename,'wb')
pickle.dump(acc_dict,outfile)
outfile.close()

acc = list(map(lambda CFOdev: list(map(lambda L: acc_dict[(CFOdev,L)], L_range)), CFOmaxdev_range))

#%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(L_range,acc[0],'*')
plt.plot(L_range, acc[1],'o')
plt.legend(['CFO Max dev = 0', 'CFO Max dev = 500'])
plt.show()
