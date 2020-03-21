import pickle
import numpy as np
import itertools

# Choose the CFO max deviation for the plot.
CFOMaxdev = 0
filelocation = "/home/vesathya/ModulationClassification/ECE257B/Models/CFOSweep/Finalresults/Resultdata/"
f = open(filelocation + "AccuracyvsCFOSweepvsSNR_CFOMaxdev" +str(CFOMaxdev) + ".pckl","rb")
acc_dict = pickle.load(f)
f.close()
fs_range = [200*pow(10,3), 1.25*pow(10,6)]
no_cfo_steps = 300
step1 = int(6e3/no_cfo_steps)
step2 = int(1000/no_cfo_steps)
marker = itertools.cycle((',', '+', '.', 'o', '*','^'))
CFO_ranges = [range(-3000, 3000, step1), range(-500,501,step2)]
# %matplotlib inline
import matplotlib.pyplot as plt
fig6 = plt.figure()
figlength = 20.5
figwidth = 12.5
fig6.set_size_inches(figlength, figwidth)
filelocation = "/home/vesathya/ModulationClassification/ECE257B/Models/CFOSweep/"
filename = "CNNoverRadioML_CFOmaxdev_" + str(CFOMaxdev) + "_L_1024__slowlearn_100EpochsRun1_model.pt"
datatypeval = "Rician"
avgwindowsize = 1
snrstep = 2
modelval = filelocation + filename
# CFO_ranges = [range(-10000, 10000, step1), range(-200,201,step2)]
if "Alldata" in modelval:
    cfoidx = 1
else:
    cfoidx = 0
cfoidx = 1
snr_range = [-10, -6, -2, 0, 2, 6, 10, 15, 20]
from statistics import mean
for SNRval in snr_range:
    accforplot = list(map(lambda cfoval:acc_dict[modelval,datatypeval,cfoval,SNRval],CFO_ranges[cfoidx]))
    accforplot_temp = list(map(lambda x: mean(accforplot[0+x:avgwindowsize+x]), range(len(accforplot)-avgwindowsize) ))
    plt.plot(CFO_ranges[cfoidx][0:len(accforplot)-avgwindowsize], accforplot_temp, markersize=2, marker=next(marker))
plt.gca().legend((snr_range), fontsize=16)
plt.title("Accuracy versus CFO sweep for different SNRs (CFO Max Dev = " + str(CFOMaxdev) + ")", fontsize=20)
plt.xlabel("CFO value (Hz)", fontsize=16)
plt.ylabel("Accuracy (%)", fontsize=16)
plt.show()