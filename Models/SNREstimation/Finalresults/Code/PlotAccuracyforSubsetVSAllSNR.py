import pickle

#f = open('/home/vesathya/ModulationClassification/ECE257B/CNN_OTARayleigh_L128_model_loss.pkl', 'rb')
f = open('/home/vesathya/ModulationClassification/ECE257B/Models/SNREstimation/acc_dictSNRall.pckl','rb')
obj = pickle.load(f)
f.close()
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt

f = open('/home/vesathya/ModulationClassification/ECE257B/Models/SNREstimation/acc_dictsubsetSNR.pckl','rb')
obj2 = pickle.load(f)
f.close()



bb = np.zeros((5,))
bb2 = np.zeros((5,))
for aa in range(5):
    bb[aa] = list(obj.values())[aa][1]
    bb2[aa] = list(obj2.values())[aa][1]
    
fig6 = plt.figure()
figlength = 18.5
figwidth = 6.5
fig6.set_size_inches(figlength, figwidth)
plt.plot(bb,'*')
plt.plot(bb2,'o')
plt.title('Accuracy versus SNR subset range, Subset ranges in SNR are: [0,2; 4,6,8; 10,12,14; 16,18,20; -20 to -2]')
plt.xlabel('SNR subset')
plt.ylabel('Accuracy')
plt.gca().legend(['SNRall','SNRsubset'])
plt.show()