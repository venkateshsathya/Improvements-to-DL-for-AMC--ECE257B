import scipy.io as sio
import numpy as np
filelocation = '/home/vesathya/ModulationClassification/ECE257B/Data/TrueCFOSweep/'
IQSamples = sio.loadmat(filelocation + 'TrueSweepRicianAWGN_CFO_0_SNR_minus20to20_nf_100_fs1pt25e6.mat')
data1 = IQSamples['dataset']
print(data1.shape)

filelocation = '/home/vesathya/ModulationClassification/ECE257B/Data/TrueCFOSweep/'
IQSamples = sio.loadmat(filelocation + 'TrueSweepRicianAWGN_CFO_0_SNR_minus20to20_nf_100_fs200e3.mat')
data0 = IQSamples['dataset']
print(data0.shape)