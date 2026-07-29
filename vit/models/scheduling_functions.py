import numpy as np
from math import exp


# We define scheduling functions to determine, at each epoch of a scheduled learner,
# the proportion of teacher forcing
# (so,unlike curriculum pacing functions going from 0 to 1, here we go from 1 to 0)
# In "Scheduled Sampling", Bengio et al. describe 3 main scheduled sampling decays :
# linear, exponential, inverse_sigmoid (for now we ignore this one)

# I construct my functions similarly to my curriculum pacing functions :
# starting from 1, lambda_f indicates the final proportion, and t_decay the time t achieve it

def linear_decay(t, lambda_f, t_decay):
	l = 1 + (lambda_f - 1) / t_decay * t
	return max(lambda_f, l)


def exponential_decay(t, lambda_f, t_decay):
	reason = lambda_f**(1/t_decay) if lambda_f > 0 else (0.01)**(1/t_decay)
	l = reason ** t
	return max(lambda_f, l)
