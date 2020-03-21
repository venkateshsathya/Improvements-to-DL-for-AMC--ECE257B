import pickle
f = open('/home/vesathya/ModulationClassification/ECE257B/Models/CFOvsL/Finalresults/Resultdata/\
SlowLearnAcc_overCFOMaxdevandL_until1408.pckl','rb')
acc_dict = pickle.load(f)
f.close()

L_range_temp =[128, 256, 512, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408]
CFOmaxdev_range = [0, 500]
acc = list(map(lambda CFOdev: list(map(lambda L: acc_dict[(CFOdev,L)], L_range_temp)), CFOmaxdev_range))

import matplotlib.pyplot as plt
plt.plot(L_range_temp,acc[0])
plt.plot(L_range_temp, acc[1])
plt.legend(['CFO Max dev = 0', 'CFO Max dev = 500'])
plt.show()
