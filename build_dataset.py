import numpy as np
import xarray as xr
import numpy as np

npz_path = "../Data/b6_traces.npz"
# ---------------------

data = np.load(npz_path)
data = data['arr_0']
n_spines, total_time = data.shape
print(f"Loaded {n_spines} spines, each with {total_time} time points")
stim_order = [150,	60,	300,	
180,	210,	120,
0,	30,	90,	
240,	270,	330,	
270,	90,	120,	
30,	240,	180,	
60,	300,	150,
330,	0,	210,
210,	270,	300,
150,	240,	90,
120,	0,	180,
60,	30,	330,
90,	240,	30,
150,	120,	300,
270,	330,	0,
180,	210,	60,
180,	150,	330,
60,	270,	210,
300,	120,	90,
0,	30,	240,
210,	240,	330,
30,	270,	60,
180,	90,	0,
300,	120,	150,
120,	30,	150,
90,	240,	210,
330,	300,	180,
0,	270,	60,
210,	180,	330,
30,	120,	240,
150,	90,	60,
270,	0,	300]
n_stims = len(stim_order)
print(f"Stimulus order has {n_stims} stims")
print(f"Remainder of total_time divided by n_stims: {total_time % n_stims}")

stim_len = total_time // n_stims 

# Reshape: (n_spines, n_stims, stim_len)
traces_3d = data.reshape(n_spines, n_stims, stim_len)

# Build xarray DataArray
da = xr.DataArray(
    traces_3d,
    dims=['spine_id', 'stim_index', 'time'],
    coords={
        'spine_id':   np.arange(n_spines),
        'stim_index': np.arange(n_stims),
        'stimulus':   ('stim_index', stim_order),   # angle for each trial
        'time':       np.arange(stim_len),
    },
    name='fluorescence'
)

print(da)
# Access by spine and stimulus angle, e.g.:
# da.sel(stim_index=da.coords['stimulus'] == 90)

# Save
da.to_netcdf('b6_spine_traces.nc')

# Load
# da = xr.open_dataarray('spine_traces.nc')
