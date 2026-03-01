import numpy as np
import xarray as xr
import numpy as np

npz_path = "../Data/anglesv2.npz"
trace_key = "traces"   # key inside the npz file
# ---------------------

data = np.load(npz_path)
traces = data[trace_key]   # shape: (n_trials, total_time)

n_trials, total_time = traces.shape
print(f"Loaded {n_trials} trials, each with {total_time} time points")
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

n_trials, total_time = traces.shape

stim_len = total_time // n_stims 

traces_reshaped = traces.reshape(
    n_trials,
    n_stims,
    stim_len
)

ds = xr.Dataset(
    data_vars=dict(
        trace=(("trial", "stim_index", "time"), traces_reshaped)
    ),
    coords=dict(
        trial=np.arange(n_trials),
        stim_index=np.arange(n_stims),
        stim=("stim_index", stim_order),
        time=np.arange(stim_len)
    ),
    attrs=dict(
        description="Stimulus-segmented traces",
    )
)

ds.to_netcdf(
    "stim_segmented_traces.nc",
    engine="netcdf4",   # optional but robust
    encoding={
        "trace": {
            "zlib": True,
            "complevel": 4
        }
    }
)

"""
Example xarray usage
ds.sel(stim=90)              # all trials, all epochs with stim=90
ds.trace.isel(trial=0)       # first trial
ds.trace.mean("trial")       # trial-averaged response

To load:
import xarray as xr

ds = xr.load_dataset("stim_segmented_traces.nc")
"""