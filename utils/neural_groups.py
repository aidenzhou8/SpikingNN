# In neural_units.py, we modeled individual neurons and synapses. But to model and manipulate data effectively, 
# we want a population of neurons, each of which responds uniquely to a given input. 
# In this file, we scale up to a true "neural network," replete with multiple neurons and synapses that update in parallel. 
# The building blocks are two classes: LIFGroup and SynapseGroup. 

import numpy as np # type: ignore

# np.random.seed(0) # Set seed for reproducibility

class LIFGroup: # A group of leaky integrate-and-fire (LIF) neurons
    def __init__(self, n=1, dim=1, t_rc=0.02, t_ref=0.002, v_th=1, 
                 max_rates=[200, 400], intercept_range=[-1, 1], t_step=0.001, v_init = 0):
        self.n = n
        # Set neuron parameters
        self.dim = dim  # Dimensionality of the input
        self.t_rc = t_rc  # Decay constant (in neurobiology, how quickly the neuron's voltage decays)
        self.t_ref = t_ref  # Refractory period
        self.v_th = np.ones(n) * v_th  # Threshold voltage for spiking
        self.t_step = t_step  # Simulation t_step
        
        # Initialize state variables
        self.voltage = np.ones(n) * v_init  # Initial voltage of neurons
        self.refractory_time = np.zeros(n)  # Tracker for refractory period
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

    def step(self, inputs):
        dt = self.t_step 

        # Update refractory time
        self.refractory_time -= dt
        delta_t = np.clip(dt - self.refractory_time, 0, dt) # ensure between 0 and dt

        # Calculate input current
        I = np.sum(self.bias + inputs * self.encoders * self.gain[:, np.newaxis], axis=1)

        # Update voltage
        self.voltage = I + (self.voltage - I) * np.exp(-delta_t / self.t_rc)

        # Check which neurons spike
        spike_mask = self.voltage > self.v_th
        self.output[:] = spike_mask / dt  # Record spikes in output

        # Calculate spike times
        t_spike = self.t_rc * np.log((self.voltage[spike_mask] - I[spike_mask]) / (self.v_th[spike_mask] - I[spike_mask])) + dt

        # Reset voltage of spiking neurons
        self.voltage[spike_mask] = 0

        # Set refractory time for spiking neurons
        self.refractory_time[spike_mask] = self.t_ref + t_spike

        return self.output  # Return the output spikes
    
class SynapseGroup:
    def __init__(self, n=1, t_s=0.05, t_step=0.001):
        self.n = n
        self.a = np.exp(-t_step / t_s)  # Decay factor for synapse
        self.b = 1 - self.a  # Scale factor for input current

        self.voltage = np.zeros(n)  # Initial voltage of neurons
    
    def step(self, inputs):
        self.voltage = self.a * self.voltage + self.b * inputs

        return self.voltage