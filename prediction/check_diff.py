import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('neuron_data_nrs.dat', 'rb') as f:
    data_old = pickle.load(f)
with open('neuron_data_nrs2.dat', 'rb') as f:
    data = pickle.load(f)

vel1 = data_old['BrainScanner20200309_153839']['velocity']
vel2 = data['BrainScanner20200309_153839']['velocity']

print("Velocity difference: "+str(np.linalg.norm(vel1 - vel2)))

vel1 = data_old['BrainScanner20200309_153839']['curvature']
vel2 = data['BrainScanner20200309_153839']['curvature']

print("Curvature difference: "+str(np.linalg.norm(vel1 - vel2)))

neur1 = data_old['BrainScanner20200309_153839']['neurons']
neur2 = data['BrainScanner20200309_153839']['neurons']

print("Neuron difference: "+str(np.linalg.norm(neur1 - neur2)))

worst = np.argmax(np.max(neur1-neur2, axis=1))
plt.plot(neur1[worst,:])
plt.plot(neur2[worst,:])
plt.show()