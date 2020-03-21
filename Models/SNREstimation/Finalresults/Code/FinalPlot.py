import pickle
f = open("/home/vesathya/ModulationClassification/ECE257B/Models/SNREstimation/Finalresults/Resultdata/acc_dictSNRall.pckl", "rb")
#ModulationClassification ECE257B Models SNREstimation Finalresults Resultdata
#f = open("acc_dictSNRall.pckl", "rb")
obj = pickle.load(f)
f.close()
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
f = open("/home/vesathya/ModulationClassification/ECE257B/Models/SNREstimation/Finalresults/Resultdata/acc_dictsubsetSNR.pckl", "rb")
obj2 = pickle.load(f)
f.close()
bb = np.zeros((5,))
bb2 = np.zeros((5,))
for aa in range(1,5):
    bb[aa] = list(obj.values())[aa-1][1]
    bb2[aa] = list(obj2.values())[aa-1][1]
bb[0] = list(obj.values())[4][1]
bb2[0] = list(obj2.values())[4][1]
fig6 = plt.figure()
figlength = 18.5
figwidth = 6.5
fig6.set_size_inches(figlength, figwidth)
plt.plot(bb, "*")
plt.plot(bb2, "o")
plt.title("Accuracy versus SNR subset range", fontsize=20)
plt.xlabel("SNR subset (dB)", fontsize=16)
plt.ylabel("Accuracy (%)", fontsize=16)
x = ["-20 to -2 ", "0,2", "4,6,8", "10,12,14", "16,18,20"]
plt.xticks(list(range(len(bb))), x)
plt.gca().legend(["SNRall","SNRsubset"], fontsize=16)
plt.show()