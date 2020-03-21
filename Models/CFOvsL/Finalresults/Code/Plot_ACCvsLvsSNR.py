## Change the CFO value to replot for 0 and 500 Hz max dev.
import pickle
filelocation = '/home/vesathya/ModulationClassification/ECE257B/Models/CFOvsL/Finalresults/Resultdata/'
f = open(filelocation + "Slowratelearn_vsCFOmaxdevvsLvsSNR_until1408.pckl","rb")
acc_dict = pickle.load(f)
f.close()
# %matplotlib inline
import matplotlib.pyplot as plt
L_range = [128, 256, 512, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408]
L_range = L_range[0::2]
L_range = [128, 256, 512, 1024, 1408]
# L_range = [128, 256, 512, 768, 832, 896, 960, 1024, 1088, 1152]
# L_range = [960, 1024, 1088, 1152]
# L_range = [128, 256, 512, 768, 960, 1024, 1152]
# L_range = [1024, 1088, 1152, 1216, 1280, 1344, 1408]
# modelrange = [ModulationPredictionCNNL128, ModulationPredictionCNNL256, ModulationPredictionCNNL512, \
#         ModulationPredictionCNNL768, ModulationPredictionCNNL1024]
CFOmaxdev_range = [500]
modulationTypes = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'PAM4', 'GFSK', 'CPFSK', 'BFM', 'DSBAM', 'SSBAM']
#acc_dict = {}
snrs = [-10, -6, -2, 0, 2, 6, 10, 15, 20]
# modelfile_location = ‘/home/vesathya/ModulationClassification/ECE257B/Models/CFOvsL/Run3_128to1024result/’
fig6 = plt.figure()
figlength = 20.5
figwidth = 20.5
fig6.set_size_inches(figlength, figwidth)
for CFOmaxdev in CFOmaxdev_range:
    if CFOmaxdev == 0:
        marker = "-o"
    else:
        marker = "-*"
    for L in L_range:
        acc_snr = list(map(lambda snr: acc_dict[CFOmaxdev, L, snr], snrs))
        #print(acc_snr)
        plt.plot(snrs, acc_snr, marker, markersize=14, label=str(CFOmaxdev) + ' ' + str(L))
plt.legend(loc= "upper left")
# plt.gca().legend(L_range)
plt.title("Accuracy vs SNR for different Input Frame Lengths (L)", fontsize=20)
plt.xlabel("SNR (dB)", fontsize=16)
plt.ylabel("Accuracy (%)", fontsize=16)
plt.ylim(0, 0.95)
plt.show()
# for snr in snrs: