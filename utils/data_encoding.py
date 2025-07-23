# This file models visual data as Poisson processes, representing values as probabilistic frequencies. 
# This matches the way that brains encode visual data - as spike trains.  

import random

# This function encodes a list of values as a spike train using a Poisson distribution. 
# The values are normalized to [0, 1], and the rates are scaled to [min_rate, max_rate]. 
# The probability of firing in each time step is calculated as the product of the rate and time step. 

# min_value and max_value represent the range of color values (default is 0-255 for grayscale images)
# min_rate and max_rate represent the range of firing rates (default is 0-10 Hz)
# dt represents the time step (default is 1ms)

# The function returns a list of booleans, where True represents a firing event and False represents no firing event. 

def poisson_fire(values, min_value=0, max_value=255, min_rate=0, max_rate=10, dt=0.001):
    relativeValues    = [ (v - min_value) / (max_value - min_value) for v in values ] # Normalize values to [0, 1]
    relativeRates     = [ min_rate + v * (max_rate - min_rate) for v in relativeValues ] # Scale rates to [min_rate, max_rate]
    probabilityOfFire = [ r * dt for r in relativeRates ] # Probability of firing at each t.s.

    firings = [random.random() < p for p in probabilityOfFire] # Generate firing events

    return firings