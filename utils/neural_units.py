# This file models "neural units": 

# FirstOrderLIF: A leaky integrate-and-fire (LIF) neuron
# FirstOrderSynapse: A post-synaptic potential (PSP) of a synapse

# Together, they are the building blocks of spiking neural networks (SNNs). 

# Leaky integrate-and-fire (LIF) neurons provide an easy way to model natural neurons. 
# An LIF neuron takes in an input (voltage or potential), which then goes through exponential decay. 
# It has a threshold potential, at which it fires,
# and a refractory period — a period of post-firing inactivity.
# The discrete, spiking nature of LIF neurons enable us to model natural cognition. 

# ALIF: An "adaptive" LIF neuron that introduces a variable threshold voltage. 
# This is useful for modeling layers of excitatory neurons.

import numpy as np # type: ignore

class FirstOrderLIF: # First Order Leaky Integrate
	def __init__(self, t_rc=0.2, t_ref=0.002, v_init=0, v_th=1): # Default values for t_rc and v_init
		self.t_rc = t_rc # speed of exponential decay
		self.v = v_init # neuron's initial potential
		self.v_th = v_th # firing threshold
		self.t_ref = t_ref # refractory period
		
		self.output = 0 # 0 if not firing, 1/T_step if firing 
		self.rt = 0 # to track refractory period
		
	def step(self, I, t_step): # Advance 1 time step (input I and time step size t_step)
		self.rt -= t_step
		
		if self.rt < 0: 
			self.v = self.v * (1 - t_step / self.t_rc) + I * t_step / self.t_rc # Integrate the input current
		
		if self.v >= self.v_th: # If voltage > threshold
			self.rt = self.t_ref
			self.output = 1 / t_step # Fire
			self.v = 0 # Reset potential
		else:  # If voltage < threshold
			self.output = 0 # Don't fire

		return self.output
	
	def reset(self): # Reset the neuron to its initial state
		self.v = self.output = self.rt = 0
	
# We use this class to model the post-synaptic potential (PSP) of a synapse. 
# Modeling the synapse, which bridges separate neurons, enables us to
# construct SNNs as graphical networks of neurons. 
# The synaptic model itself is very akin to that of the LIF neuron
# (which is intuitive, as the post-synaptic potential just encodes neuron output). 

class FirstOrderSynapse: 
    def __init__(self, t_s=0.01):
        self.t_s  = t_s # Synaptic time constant
        self.output = 0     # Current potential

    def step(self, I, t_step):
        self.output = self.output * (1 - t_step / self.t_s) + I * t_step / self.t_s # Decay potential
        return self.output
    
    def reset(self): # Reset the synapse to its initial state
        self.output = 0

"""
neuron  = FirstOrderLIF(t_rc=0.02, t_ref=0.2)
synapse = FirstOrderSynapse(t_s=0.2)

# A function that advances both the neuron and synapse by 1 time step
def combinedStep(I, t_step):
	neuron_output  = neuron.step(I, t_step)
	synapse_output = synapse.step(neuron_output, t_step)
	return synapse_output
"""

# This class is mostly a copy of the FirstOrderLIF class, but with an additional variable "inh" -- the inhibitory current.
class ALIF:
    def __init__(self, n=1, dim=1, t_rc=0.02, t_ref=0.002, v_th=1, 
                 max_rates=[200, 400], intercept_range=[-1, 1], t_step=0.001, v_init = 0,
                 t_inh=0.05, inc_inh=1.0 
                 ):
        self.n = n
        # Set parameters
        self.dim = dim  # Dimensionality of the input
        self.t_rc = t_rc  # Decay constant
        self.t_ref = t_ref  # Refractory period
        self.v_th = np.ones(n) * v_th  # Threshold voltage
        self.t_step = t_step  # Time step for simulation

        self.inh = np.zeros(n)  
        self.t_inh = t_inh 
        self.inc_inh = inc_inh 
        
        # Initialize state variables
        self.voltage = np.ones(n) * v_init  # Initial voltage of neurons
        self.refractory_time = np.zeros(n)  # Time remaining in refractory period
        self.output = np.zeros(n)  # Output spikes

        # Generate max rates and intercepts within the given range
        max_rates_tensor = np.random.uniform(max_rates[0], max_rates[1], n)
        intercepts_tensor = np.random.uniform(intercept_range[0], intercept_range[1], n)

        # Calculate gain and bias for each neuron
        self.gain = self.v_th * (1 - 1 / (1 - np.exp((self.t_ref - 1/max_rates_tensor) / self.t_rc))) / (intercepts_tensor - 1)
        self.bias = np.expand_dims(self.v_th - self.gain * intercepts_tensor, axis=1)
        
        # Initialize encoders
        self.encoders = np.random.randn(n, self.dim)
        self.encoders /= np.linalg.norm(self.encoders, axis=1)[:, np.newaxis]

    def reset(self):
        # Reset the state variables to initial conditions
        self.voltage = np.zeros(self.n)
        self.refractory_time = np.zeros(self.n)
        self.output = np.zeros(self.n)
        self.inh = np.zeros(self.n)

    def step(self, inputs):
        dt = self.t_step 

        # Update refractory time
        self.refractory_time -= dt
        delta_t = np.clip(dt - self.refractory_time, 0, dt) # ensure between 0 and dt

        # Calculate input current
        I = np.sum(self.bias + inputs * self.encoders * self.gain[:, np.newaxis], axis=1)

        # Update decay constant
        self.voltage = I + (self.voltage - I) * np.exp(-delta_t / self.t_rc)

        # Check for spiking neurons
        spike_mask = self.voltage > self.v_th + self.inh
        self.output[:] = spike_mask / dt  # Record spikes in output

        # Calculate spike timing
        t_spike = self.t_rc * np.log((self.voltage[spike_mask] - I[spike_mask]) / (self.v_th[spike_mask] - I[spike_mask])) + dt

        # Reset voltage of spiking neurons
        self.voltage[spike_mask] = 0

        # Set refractory period
        self.refractory_time[spike_mask] = self.t_ref + t_spike

        self.inh = self.inh * np.exp(-dt / self.t_inh) + self.inc_inh * (self.output > 0)  # <--- ADDED

        return self.output  # Return the output spikes