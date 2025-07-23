# This file plots the output of a synapse for a given input signal.
# The given input signal is a square wave with a frequency of 5 Hz.

# This single neuron-synapse model "encodes" the input signal as a spike train. 
# To decode it, we can use an analytical method, but it's not very accurate. 
# This is the motivation for using a "network" of LIF neurons. 

import math
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from scipy import signal as sg # type: ignore
import sys
sys.path.append("/Users/AidenZhou/Desktop/CS/Projects/SpikingNN/") # obviously, change this to your own path
from utils.neural_units import FirstOrderLIF, FirstOrderSynapse

def signal(t): # Function for input signal
    return 4*sg.square(2 * np.pi * 5 * t)

# Init LIF neuron, and synapse with synaptic constant = 0.15
neuron = FirstOrderLIF()
synapse = FirstOrderSynapse(t_s = 0.15)

t_step = 0.01
t = np.arange(0, 12, t_step)
signal_out = []
synapse_out = []
for i in t:
    sig = signal(i)
    neuron.step(sig, t_step)
    synapse.step(neuron.output, t_step)
    signal_out.append(sig)
    synapse_out.append(synapse.output)

# Create a single figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot input signal on the left subplot
ax1.plot(t, signal_out, color="C1")
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Signal')
ax1.set_title('Input Signal')

# Plot synapse output on the right subplot
ax2.plot(t, synapse_out)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Output')
ax2.set_title('Synapse Output')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()