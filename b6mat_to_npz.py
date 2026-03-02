import scipy.io
import numpy as np

mat = scipy.io.loadmat('../Data/B6_neuron2.mat', simplify_cells=True)

np.savez('b6_traces.npz',mat['snips'].T)
