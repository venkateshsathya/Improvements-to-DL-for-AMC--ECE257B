import pickle
import matplotlib.pyplot as plt

filelocation = '/home/vesathya/ModulationClassification/ECE257B/Models/CFOvsL/Finalresults/Resultdata/'
f = open(filelocation + "SlowLearnAcc_overCFOMaxdevandL_avgoverSNR_until1408.pckl","rb")
acc_dict = pickle.load(f)
f.close()
L_range_temp =[128, 256, 512, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408]
CFOmaxdev_range = [0, 500]
acc = list(map(lambda CFOdev: list(map(lambda L: acc_dict[(CFOdev,L)], L_range_temp)), CFOmaxdev_range))

avg_cfo_0_moving4 = list()
for i in range(0, len(acc[0])-3):
    avg_cfo_0_moving4.append(sum(acc[0][i:i+4])/4)

# print(avg_cfo_0_moving4)

avg_cfo_500_moving4 = list()
for i in range(0, len(acc[1])-3):
    avg_cfo_500_moving4.append(sum(acc[1][i:i+4])/4)

# print(avg_cfo_500_moving4)

avg_cfo_0_moving2 = list()
for i in range(0, len(acc[0])-1):
    avg_cfo_0_moving2.append(sum(acc[0][i:i+2])/2)

# print(avg_cfo_0_moving2)

avg_cfo_500_moving2 = list()
for i in range(0, len(acc[1])-1):
    avg_cfo_500_moving2.append(sum(acc[1][i:i+2])/2)

# print(avg_cfo_500_moving2)

avg_cfo_0_moving3 = list()
for i in range(0, len(acc[0])-2):
    avg_cfo_0_moving3.append(sum(acc[0][i:i+3])/3)

# print(avg_cfo_0_moving3)

avg_cfo_500_moving3 = list()
for i in range(0, len(acc[1])-2):
    avg_cfo_500_moving3.append(sum(acc[1][i:i+3])/3)

# print(avg_cfo_500_moving3)

avg_cfo_0_moving5 = list()
for i in range(0, len(acc[0])-4):
    avg_cfo_0_moving5.append(sum(acc[0][i:i+5])/5)

# print(avg_cfo_0_moving5)

avg_cfo_500_moving5 = list()
for i in range(0, len(acc[1])-4):
    avg_cfo_500_moving5.append(sum(acc[1][i:i+5])/5)

# print(avg_cfo_500_moving5)

fig6 = plt.figure()
figlength = 20.5
figwidth = 10.5
fig6.set_size_inches(figlength, figwidth)
plt.plot(L_range_temp, acc[0])
plt.plot(L_range_temp, acc[1])
plt.legend(["CFO Max dev = 0", "CFO Max dev = 500"], fontsize=16)
plt.title("Accuracy plot with Input Frame Length (L)", fontsize=20)
plt.xlabel("Input Frame Length", fontsize=16)
plt.ylabel("Accuracy (%)", fontsize=16)
plt.show()

# fig6 = plt.figure()
# figlength = 20.5
# figwidth = 10.5
# fig6.set_size_inches(figlength, figwidth)
# plt.plot(range(0, len(avg_cfo_0_moving2)), avg_cfo_0_moving2)
# plt.plot(range(0, len(avg_cfo_500_moving2)), avg_cfo_500_moving2)
# plt.legend(["CFO Max dev = 0", "CFO Max dev = 500"])
# plt.title("Plot")
# plt.xlabel("Window")
# plt.ylabel("%")
# plt.show()
# plt.plot(range(0, len(avg_cfo_0_moving2)), [a2 - b2 for a2, b2 in zip(avg_cfo_0_moving2, avg_cfo_500_moving2)])
# plt.ylim(0, 0.1)
# plt.show()
#
# plt.plot(range(0, len(avg_cfo_0_moving3)), avg_cfo_0_moving3)
# plt.plot(range(0, len(avg_cfo_500_moving3)), avg_cfo_500_moving3)
# plt.show()
# plt.legend(["CFO Max dev = 0", "CFO Max dev = 500"])
# plt.title("Plot")
# plt.xlabel("Window")
# plt.ylabel("%")
# plt.plot(range(0, len(avg_cfo_0_moving3)), [a3 - b3 for a3, b3 in zip(avg_cfo_0_moving3, avg_cfo_500_moving3)])
# plt.ylim(0, 0.1)
# plt.show()
#
# plt.plot(range(0, len(avg_cfo_0_moving4)), avg_cfo_0_moving4)
# plt.plot(range(0, len(avg_cfo_500_moving4)), avg_cfo_500_moving4)
# plt.show()
# plt.legend(["CFO Max dev = 0", "CFO Max dev = 500"])
# plt.plot(range(0, len(avg_cfo_0_moving4)), [a4 - b4 for a4, b4 in zip(avg_cfo_0_moving4, avg_cfo_500_moving4)])
# plt.ylim(0, 0.1)
# plt.show()

plt.plot(range(0, len(avg_cfo_0_moving5)), avg_cfo_0_moving5)
plt.plot(range(0, len(avg_cfo_500_moving5)), avg_cfo_500_moving5)
plt.legend(["CFO Max dev = 0", "CFO Max dev = 500"], fontsize=14)
plt.title("Accuracy plot averaged across 5 values", fontsize=16)
plt.xlabel("Moving window indexes", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.show()

plt.plot(range(0, len(avg_cfo_0_moving5)), [a5 - b5 for a5, b5 in zip(avg_cfo_0_moving5, avg_cfo_500_moving5)])
plt.ylim(0, 0.1)
plt.title("Difference in accuracies for 2 CFO values", fontsize=16)
plt.xlabel("Moving window indexes", fontsize=14)
plt.ylabel("Difference in Accuracy", fontsize=14)
plt.show()

# fig6 = plt.figure()
# figlength = 20.5
# figwidth = 10.5
# fig6.set_size_inches(figlength, figwidth)




# # plt.legend(["CFO Max dev = 0", "CFO Max dev = 500"])
