import pickle
import numpy as np

CFOMaxdev = 0
filename = '/home/vesathya/ModulationClassification/ECE257B/Data/CFOvsL/CFO_'+\
    str(CFOMaxdev) +'_L1024.pckl'
    
#ModulationClassification ECE257B Data CFOvsL
print("Dataset Filename is:  ", filename)
f = open(filename,'rb')
dataset = pickle.load(f,encoding='latin1')
f.close()


snrs,mods = map(lambda j: sorted(list  (set  (map(lambda x: x[j], dataset.keys())  )  )), [1,0])
data = np.empty((1,len(snrs),1100,2,1024))
snridx = 0
for snr in snrs:
    X = []
    for mod in mods:
        temp = dataset[(mod,snr)]
        X.append(temp[0:100,:,:])
        
    X = np.vstack(X)
    data[0,snridx,:,:] = X
    snridx = snridx + 1
label_dict = {'QAM16':0, 'QAM64':1, '8PSK':2, 'WBFM':3, 'BPSK':4, 'CPFSK':5, 'AM-DSB':6, 'GFSK':7,
     'PAM4':8, 'QPSK':9, 'AM-SSB':10}

label = np.empty((1100,))
modidx = 0
for mod in mods:
    print(label_dict[mod])
    label[modidx*100:(modidx+1)*100] = label_dict[mod]*np.ones((100,))
    modidx = modidx + 1

print(label)