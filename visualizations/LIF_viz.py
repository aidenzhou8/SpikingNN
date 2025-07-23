# This file plots the output of a LIF neuron. 

import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import sys
sys.path.append("/Users/AidenZhou/Desktop/CS/Projects/SpikingNN/") # obviously, change this to your own path
from utils.neural_units import FirstOrderLIF
from scipy import signal as sg # type: ignore

# Simulation parameters
v_th    = 1.000 # Threshold voltage
t_ref = 0.200 # Refractory period 
t_rc  = 0.10  # Decay constant

duration = 6   # Duration of simulation
T_step = 0.001 # Time step size

times = np.arange(1, duration, T_step) # Create a range of time values

neuron = FirstOrderLIF(t_rc=t_rc, t_ref=t_ref, v_th=v_th) # Create a new LIF neuron

def signal(t): # Function for input signal
    return 2*sg.square(2 * np.pi * 3 * t)

I_history = []
v_history = []
output_history = []
vth_history = []

for t in times: # Iterate over each time step
    I = signal(t)     # Get the input current at this time
    neuron.step(I, T_step) # Advance the neuron one time step

    I_history.append(I)    # Record the input current
    v_history.append(neuron.v) # Record the neuron's potential
    output_history.append(neuron.output * T_step / 10) # Record the neuron's output (scaled)
    vth_history.append(neuron.v_th) # Record the neuron's threshold


plt.figure() # Create a new figure
plt.plot(times, I_history, color="grey", linestyle="--")
plt.plot(times, vth_history, color="green", linestyle="--")
plt.plot(times, v_history)
plt.plot(times, output_history, color="red", linewidth=2.5)
plt.xlabel('Time (s)') # Label the x-axis
plt.legend(['Input current', 'neuron.v_th', 'neuron.v', 'neuron.output']) # Add a legend
plt.show() # Display the plot