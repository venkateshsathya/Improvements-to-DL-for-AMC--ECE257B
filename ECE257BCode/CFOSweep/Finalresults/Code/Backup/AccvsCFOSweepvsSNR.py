import pickle
import numpy as np
import itertools
filelocation = '/home/vesathya/ModulationClassification/ECE257B/Models/CFOSweep/'
f = open(filelocation + 'Accuracydict_trueCFOSweep13_03_2020_04_28_58.pckl','rb')
acc_dict = pickle.load(f)
f.close()
     
fs_range = [200*pow(10,3), 1.25*pow(10,6)]
no_cfo_steps = 300
step1 = int(6e3/no_cfo_steps)
step2 = int(1000/no_cfo_steps)

marker = itertools.cycle((',', '+', '.', 'o', '*','^')) 
# model_file1 = modelfilelocation + 'CNN_Alldata.pt'
# model_file2 = modelfilelocation + 'Resnet_Alldata.pt'
# model_file3 = modelfilelocation + 'Resnet_model_ece228.pt'
# model_file4 = modelfilelocation + 'CNN_model_ece228.pt'

CFO_ranges = [range(-3000, 3000, step1), range(-500,501,step2)]

%matplotlib inline
import matplotlib.pyplot as plt
fig6 = plt.figure()
figlength = 20.5
figwidth = 16.5
fig6.set_size_inches(figlength, figwidth)

filelocation = '/home/vesathya/ModulationClassification/ECE257B/Models/CFOSweep/'
filename = 'CNNoverRadioML_CFOmaxdev_500_L_1024__slowlearn_100EpochsRun1_model.pt'
datatypeval = 'Rician'
avgwindowsize = 1
snrstep = 2

modelval = filelocation + filename
# CFO_ranges = [range(-10000, 10000, step1), range(-200,201,step2)]

if 'Alldata' in modelval:
    cfoidx = 1
else:
    cfoidx = 0   

cfoidx = 1
snr_range = [-10, -6, -2, 0, 2, 6, 10, 15, 20]
from statistics import mean
for SNRval in snr_range:
    accforplot = list(map(lambda cfoval:acc_dict[modelval,datatypeval,cfoval,SNRval],CFO_ranges[cfoidx]))
    accforplot_temp = list(map(lambda x: mean(accforplot[0+x:avgwindowsize+x]), \
                               range(len(accforplot)-avgwindowsize) ))
    plt.plot(CFO_ranges[cfoidx][0:len(accforplot)-avgwindowsize], \
             accforplot_temp,markersize=6, marker = next(marker))
    
plt.gca().legend((snr_range))
plt.show()
