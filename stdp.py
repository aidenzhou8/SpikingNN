# To enable our SNN to learn, we write a class that models spike-timing-dependent plasticity (STDP), a kind of Hebbian learning rule. 
# STDP works by updating the weights of synapses based on the firing patterns of pre- and post-synaptic neurons. 
# For instance, if neuron A consistently fires right before neuron B, the weight of the synapse between A and B is increased. 
# Note that order (pre-post) very much matters here. This model of natural learning can be further refined 
# by using a sophisticated model of synaptic plasticity (pre-post-pre vs naive pre-post). 

import numpy as np # type: ignore

class STDPWeights:
    def __init__(self, nPre, nPost, t_plus = 0.02, t_minus = 0.02, a_plus = 0.01, a_minus = 0.011, g_min=0, g_max=1):
        self.nPre = nPre # # of pre-synaptic neurons
        self.nPost = nPost # # of post-synaptic neurons
        self.t_plus = t_plus # Time constant for positive STDP
        self.t_minus = t_minus # Time constant for negative STDP
        self.a_plus = a_plus # Learning rate for positive STDP
        self.a_minus = a_minus # Learning rate for negative STDP
        self.x = np.zeros(nPre) # Trace for pre-synaptic neurons
        self.y = np.zeros(nPost) # Trace for post-synaptic neurons

        self.g_min = g_min # Min weight
        self.g_max = g_max # Max weight
        self.w = np.zeros((nPre, nPost)) # A 2-D array; nPre x nPost zeros (typical is random)


    def step(self, t_step):
        self.x = self.x * np.exp(-t_step/self.t_plus) # Decay trace for pre-synaptic neurons
        self.y = self.y * np.exp(-t_step/self.t_minus) # Decay trace for post-synaptic neurons

    # Update synaptic weights based on firing patterns of the pre- and post-synaptic neurons
    def updateWeights(self, preOutputs, postOutputs):
        self.x += (preOutputs  > 0) * self.a_plus 
        self.y -= (postOutputs > 0) * self.a_minus

        alpha_g = self.g_max - self.g_min # Scaling factor for weight updates

        preSpikeIndices = np.where(preOutputs > 0)[0]   # Indices of pre-synaptic neurons
        postSpikeIndices = np.where(postOutputs > 0)[0] # Indices of post-synaptic neurons

        for ps_idx in preSpikeIndices:
            self.w[ps_idx] += alpha_g * self.y
            self.w[ps_idx] = np.clip(self.w[ps_idx], self.g_min, self.g_max)

        for ps_idx in postSpikeIndices:
            self.w[:, ps_idx] += alpha_g * self.x
            self.w[:, ps_idx] = np.clip(self.w[:, ps_idx], self.g_min, self.g_max)