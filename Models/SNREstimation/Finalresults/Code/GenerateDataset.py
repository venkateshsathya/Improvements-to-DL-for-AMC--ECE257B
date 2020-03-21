
import numpy as np
import pickle
import datetime
from ModulationPredictionResNet import ModulationPredictionResNet
from ModulationPredictionCNN import ModulationPredictionCNN
import torch

SNRLevels = list(range(20,-21,-2))
modulationTypes = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", "BFM", "DSBAM", "SSBAM"];
label_dict = {'16QAM':0, '64QAM':1, '8PSK':2, 'BFM':3, 'BPSK':4, 'CPFSK':5, 'DSBAM':6, 'GFSK':7,
     'PAM4':8, 'QPSK':9, 'SSBAM':10}

def loaddata(datafilename, labelfilename):
    data = np.empty((0,2,1024))
    NumFiles = 5 # We are splitting the data into 5 files since a single pickle file cannot exceed say 4 GB in size
    for i in range(0,NumFiles+1):
        filename = datafilename + '_' + str(i)
        infile = open(filename,'rb')
        data = np.append(data, pickle.load(infile),axis=0)
        print(data.shape)
        infile.close()
    infile = open(labelfilename,'rb')
    label = pickle.load(infile)
    print(label.shape)
    infile.close()
    
    return data, label

def generateDatabySNR(data):
    
    sizeperSNR = 55000
    spf = 1024
    sizeperSNRperMod = 5000
    sizeperMod = 5000*len(SNRLevels)
    # Creating a data format - where we can straight away extract all data for each SNR type
    databySNR = np.zeros((len(SNRLevels),sizeperSNR,2,spf)) 
    #LabelbySNR2 = np.zeros((sizeperSNR,), dtype=int)
    for SNRidx in range(len(SNRLevels)):
        for modidx in range(len(modulationTypes)):
            databySNR[SNRidx, modidx*sizeperSNRperMod:(modidx+1)*sizeperSNRperMod,:,:] = \
            data[SNRidx*sizeperSNRperMod + modidx*sizeperMod :(SNRidx+1)*sizeperSNRperMod + \
                 modidx*sizeperMod]

    # Creating labels for the data -- note that the labels hold good for all SNR levels
    LabelbySNR = 30*np.ones((sizeperSNR,), dtype=int)# we are setting a value 30 - not equal to any of the valid labels.
    modidx = 0
    for modval in modulationTypes:
        LabelbySNR[modidx*sizeperSNRperMod : (modidx+1)*sizeperSNRperMod] = \
        label_dict[modval]*np.ones((sizeperSNRperMod,), dtype=int)
        modidx = modidx+1
    
    return databySNR, LabelbySNR

datafilename = '/home/vesathya/ModulationClassification/IQSamples/ICAASP2020Data/PickledData/Rayleigh'
labelfilename = datafilename + '_Labels'
dataRayleigh, labelRayleigh = loaddata(datafilename, labelfilename)
databySNRRayleigh, labelbySNRRayleigh = generateDatabySNR(dataRayleigh)

## Trim it to length L = 128
L = 128
dataRayleighL128 = databySNRRayleigh[:,:,:,0:L]
print(dataRayleighL128.shape)

data = dataRayleighL128
label = labelbySNRRayleigh
print(data.shape, label.shape)

## Subset ranges in SNR are: [0,2; 4,6,8; 10,12,14; 16,18,20] - lowest subset has one region lesser - 
## harder to learn for lower SNR.
### I have swapped the SNR regions of subset 1 and 4 to verify something. CHange it back. 
### Values swapped back - it was used tofind a bug where the range which is origianlly 
### from 20 to -20 was though of as gfoing from -20 to 20

# Dictionary of SNR subsets
databySNRsubsetdict = {}

SNRregions = {'SNRsubset1': [0,2], 'SNRsubset2': [4,6,8],'SNRsubset3': [10,12,14],'SNRsubset4': [16,18,20] }

for snrsubset in SNRregions.keys():
    
    # 0 SNR corresponds to index of 10(11th value). Therefore (-SNR)/2 + 10 is the correct mapping of SNR to
    # indexing within the context of 20:-20:-2, where 20 corresponds to 0 and -20 to 20 and 0 to 10.
    
    num_snr_subset = len(SNRregions[snrsubset])
    temp2 = data[(-1*np.array(SNRregions[snrsubset])/2 +10).astype(int)]

    databySNRsubsetdict[snrsubset + '_data'] = temp2.reshape((-1,2,L))
    databySNRsubsetdict[snrsubset + '_label'] = np.array(list(label[:])*num_snr_subset)

## Note that regarding label - the label is the same even if we truncate a subset of the SNR.