import scipy.io
import numpy as np
from scipy.interpolate import interp1d

mat = scipy.io.loadmat('Liam_data.mat', simplify_cells=True)
C = mat['C']

# --- Parameters ---
trace_type = 'fulltrace'    # 'Snips' or 'fulltrace'
mode = 'grey'          # 'angle' or 'grey' (only used if trace_type == 'Snips')

all_traces = []
all_ids    = []

for recording in C:
    d_ids = recording['dID']

    if trace_type == 'Snips':
        snips = recording['Snips']
        for spine_struct, spine_id in zip(snips, d_ids):
            trace = spine_struct['spine'][mode]
            all_traces.append(np.array(trace).flatten())
            all_ids.append(spine_id)
    else:  # fulltrace is a 2D matrix, each row is a spine trace
        fulltrace = np.array(recording['fulltrace'])
        for trace, spine_id in zip(fulltrace, d_ids):
            all_traces.append(trace.flatten())
            all_ids.append(spine_id)

# Check length distribution
lengths = [len(t) for t in all_traces]
print(f"Trace lengths — Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.0f}")

# Downsample to shortest trace
def downsample_trace(trace, target_length):
    x_old = np.linspace(0, 1, len(trace))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, trace, kind='linear')
    return f(x_new)

min_len = min(lengths)
resampled = np.vstack([
    downsample_trace(t, min_len) if len(t) > min_len else t
    for t in all_traces
])

ids = np.array(all_ids)

print(f"Final array shape: {resampled.shape}")

np.savez('spines.npz', ids=ids, traces=resampled)
