# Spiking Neural Network

I made a basic spiking neural network (SNN) to simulate bio-plausible cognitive behavior on conventional hardware. This project illustrates how SNNs, which utilize discrete "spikes" to store and analyze information, can achieve energy-efficient calculations while delivering strong results on the MNIST and CIFAR-100 datasets. 

## Overview

Spiking neural networks (SNNs) represent a potential major shift in the ML industry. They have proven an efficiency advantage over traditional artificial neural networks by copying the discrete nature of biological neurons. Unlike naive neurons that continuously take in values, the leaky integrate-and-fire (LIF) neuron model only spikes when certain conditions are met. This is part of why natural cognition is not nearly as energy-hungry as AI, but can achieve superior results in various aspects. If advances in biology-inspired hardware can catch up (for instance, IBM's TrueNorth chip), the future is bright for SNNs.

This repo includes code for:
- **Basic neural units**: LIF neurons and synapses
- **Population coding**: Groups of neurons for robust and parallelizable representation
- **Learning mechanisms**: Via STDP (spike-timing-dependent plasticity)
- **Data encoding**: Poisson encoding for visual stimuli
- **Visualizations**: For single and groups of LIF neurons, synapses, and Poisson encoding. 

## Project Structure

```
SpikingNN/
├── utils/                    
│   ├── neural_units.py      # Single LIF neurons and synapses
│   ├── neural_groups.py     # Population coding with LIFGroup and SynapseGroup
│   ├── data_encoding.py     # Poisson encoding for visual data
│   └── __init__.py
├── visualizations/           
│   ├── LIF_viz.py          # LIF neuron visualization
│   ├── synapse_viz.py      # Synaptic response visualization
│   ├── encoding_viz.py     # Poisson encoding visualization
│   └── __init__.py
├── data/                    
│   └── brain_icon.png     
├── stdp.py                  # STDP learning 
└── README.md
```

## Key Features

### 🧠 Neural Units
- **FirstOrderLIF**: Leaky integrate-and-fire neuron
- **FirstOrderSynapse**: Post-synaptic potential modeling
- **ALIF**: Adaptive leaky integrate-and-fire neuron (variable spiking threshold)
- **LIFGroup**: Population of neurons with diverse tuning curves
- **SynapseGroup**: Network unifying a neuron population

### 📊 Data Encoding
- **Poisson Encoding**: Converts continuous values to spike trains
- **Population Coding**: Distributed representation over multiple neurons
- **Visual Data Conversion**: Arbitrary picture-to-spike-train conversion: CV for SNNs

### 🎓 Learning Mechanisms
- **STDP**: Biology-inspired Hebbian learning rule (neurons that fire together, wire together)
- **Weight Adaptation**: Synaptic strength modification based on consecutively spiking neurons

### 📈 Visualization Tools
- **LIF Dynamics**: Decay curves, spikes, and refractory periods
- **Encoding Visualization**: Original pictures vs. spike-based reconstructions

## Usage Guide

```bash
# Clone the repository
git clone https://github.com/aidenzhou8/SpikingNN.git
cd SpikingNN

# Get dependencies
pip install numpy matplotlib scipy pillow
```

## Tips and Tricks

### Run LIF Neuron Visualization
```bash
python visualizations/LIF_viz.py
```

### Run Synapse Visualization
```bash
python visualizations/synapse_viz.py
```

### Run Data Encoding Visualization
```bash
python visualizations/encoding_viz.py
```

## Credits

This repo is mainly inspired by:
- Steve Oney's "Building Spiking Neural Networks (SNNs) from Scratch"
- "Neuronal Dynamics" by Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski
