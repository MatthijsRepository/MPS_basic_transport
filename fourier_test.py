import numpy as np
import pickle
import os
os.chdir("C:\\Users\\matth\\OneDrive\\Documents\\TUDelft\\MEP\\code\\MPS_basic_transport")

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def load_state(folder, name, new_ID):
    """ loads a pickled state from folder 'folder' with name 'name' - note: name must include .pkl """
    filename = folder + name
    with open(filename, 'rb') as file:  
        loaded_state = pickle.load(file)
    globals()[loaded_state.name] = loaded_state
    
    loaded_state.ID = new_ID
    if loaded_state.is_density:
        loaded_state.name = "DENS"+str(new_ID)
    else: 
        loaded_state.name = "MPS"+str(new_ID)
    return loaded_state





loadstate_folder = "data\\"
#loadstate_filename = "1220_1546_DENS1_N20_chi25.pkl"
#loadstate_filename = "1221_1441_DENS1_N21_chi25.pkl"  #NORM: 0.9392

loadstate_filename = "0103_0926_DENS1_N21_chi35.pkl"

DENS1 = load_state(loadstate_folder, loadstate_filename, 1)



#DENS1.spin_current_values[4999] *= -1
#DENS1.store()


#"""
plt.plot(DENS1.normalization)
plt.title("Normalization")
plt.grid()
plt.show()

plt.plot(DENS1.spin_current_values)
plt.grid()
plt.show()


print(len(DENS1.spin_current_values))
print(np.pi**2 / DENS1.N**2)
#DENS1.spin_current_values /= 0.944459559

plt.plot(DENS1.spin_current_values[-28000:])
plt.grid()
plt.show()



filter_start = -28000
x_data = np.arange(0, -80-filter_start)


a = DENS1.spin_current_values[filter_start:]
avg = np.average(a)
a = a-avg


wow = np.fft.rfft(a)
plt.plot(wow[:1000])
plt.show()


low_lim = 250
up_lim=1000
#wow[58] = (wow[59]+wow[57])/2
#wow[61] = (wow[62]+wow[60])/2
plt.plot(np.arange(low_lim,up_lim), wow[low_lim:up_lim])
wow[low_lim:up_lim] = (wow[low_lim] + wow[up_lim])/2
plt.plot(np.arange(low_lim,up_lim), wow[low_lim:up_lim], color="red")

plt.show()


b = np.fft.irfft(wow)+avg
#plt.plot(b)
#plt.show()





plt.plot(DENS1.spin_current_values[filter_start:])
plt.plot(b, color="red")
plt.grid()
plt.show()

#"""

"""
def functional(x, a, b, c, d):
    return b*np.exp(a*x) - c*(d*x+1)**1/2

print()
print("here")
print()
fit_params = curve_fit(functional, x_data, b[40:-40], p0=np.array([-0.00003574, 0.0125, 0.0125,1000]))

pa, pb, pc, pd = fit_params[0][0], fit_params[0][1], fit_params[0][2], fit_params[0][3]

print(pa, pb, pc, pd)


fit = functional(x_data, pa, pb, pc, pd)

plt.plot(b[40:-40])
plt.plot(fit)
plt.grid()
plt.show()
"""


