import numpy as np
import scipy.io
import xarray as xr

# ── Load .mat file ────────────────────────────────────────────────────────────
mat = scipy.io.loadmat('Data/b6_neuron.mat', simplify_cells=True)
snp = mat['SNP2']  # length-87 array of structs

# ── Trial metadata ────────────────────────────────────────────────────────────
N_TRIALS = 96

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
assert len(stim_order) == N_TRIALS

trial_idx  = np.arange(N_TRIALS)
stim_array = np.array(stim_order, dtype=np.int16)

# ── Helper: reshape (total_frames × n_rois) → (n_trials × frames_per_trial × n_rois)
def reshape_to_trials(data: np.ndarray, n_trials: int) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, np.newaxis]
    total_frames, n_rois = data.shape
    assert total_frames % n_trials == 0, (
        f"total_frames ({total_frames}) not evenly divisible by n_trials ({n_trials})"
    )
    frames_per_trial = total_frames // n_trials
    return data.reshape(n_trials, frames_per_trial, n_rois)

# ── Accumulate ROIs across all 87 dendrite groups ────────────────────────────
# For each field (angle, grey) we build lists across every individual ROI,
# tracking structure identity and parent dendrite index.
# Fields are handled separately since frames_per_trial differs between them.

fields     = ('angle', 'grey')
structures = ('dend', 'spine', 'shaft')

accum = {f: {'data': [], 'structure': [], 'parent_dend': []} for f in fields}

for dend_idx, row in enumerate(snp):
    for struct_name in structures:
        struct = row[struct_name]

        for field in fields:
            raw      = np.array(struct[field], dtype=np.float32)  # (total_frames, n_rois)
            reshaped = reshape_to_trials(raw, N_TRIALS)            # (96, fpT, n_rois)
            n_rois   = reshaped.shape[2]

            for roi_i in range(n_rois):
                accum[field]['data'].append(reshaped[:, :, roi_i])  # (96, fpT)
                accum[field]['structure'].append(struct_name)
                accum[field]['parent_dend'].append(dend_idx)

# ── Build one DataArray per field, then merge into a Dataset ─────────────────
data_arrays = {}

for field in fields:
    acc          = accum[field]
    data_stack   = np.stack(acc['data'], axis=-1)  # (96, fpT, n_rois_total)
    n_rois_total     = data_stack.shape[2]
    frames_per_trial = data_stack.shape[1]

    da = xr.DataArray(
        data_stack,
        dims=['trial', 'frame', 'roi'],
        coords={
            'trial'          : trial_idx,
            'stimulus_angle' : ('trial', stim_array),
            'frame'          : np.arange(frames_per_trial),
            'roi'            : np.arange(n_rois_total),
            'structure'      : ('roi', np.array(acc['structure'])),
            'parent_dendrite': ('roi', np.array(acc['parent_dend'], dtype=np.int16)),
        },
        name=field,
        attrs={
            'field'           : field,
            'units'           : 'degrees' if field == 'angle' else 'a.u.',
            'frames_per_trial': int(frames_per_trial),
            'n_trials'        : N_TRIALS,
            'n_rois_total'    : int(n_rois_total),
        },
    )
    data_arrays[field] = da

# ── Assemble Dataset ──────────────────────────────────────────────────────────
ds = xr.Dataset(
    data_arrays,
    attrs={
        'description'      : 'SNP2 neural imaging data — all ROIs unified, indexed by trial',
        'n_trials'         : N_TRIALS,
        'n_dendrite_groups': len(snp),
        'stim_order'       : ','.join(map(str, stim_order)),   # stored as "150,60,300,..."
        'structures'       : ','.join(structures),             # stored as "dend,spine,shaft"
        'notes'            : (
            'roi coords: structure (dend/spine/shaft), parent_dendrite (0-86). '
            'Dendrite ROIs are self-referential (parent_dendrite == their group index).'
        ),
    },
)

print(ds)

# ── Example selections ────────────────────────────────────────────────────────
# All spine ROIs:
#   spine_rois = ds.roi.where(ds.structure == 'spine', drop=True)
#   ds['grey'].sel(roi=spine_rois)
#
# All ROIs belonging to dendrite group 3 (its own dendrite + child spines/shafts):
#   group3_rois = ds.roi.where(ds.parent_dendrite == 3, drop=True)
#   ds['grey'].sel(roi=group3_rois)
#
# Trials at 90° stimulus:
#   ds['grey'].where(ds.stimulus_angle == 90, drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = 'Data/b6_neuron_snp2.nc'
ds.to_netcdf(out_path, engine='netcdf4')
print(f"\nSaved to {out_path}")