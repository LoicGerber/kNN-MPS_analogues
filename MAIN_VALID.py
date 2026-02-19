#%%
import tifffile as tf
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

from kNN_FUNC import prepare_knn_data, normalize_predictors, knn_analogue_selection, knn_image_generation
from QS_FUNC import prepare_TIs_DI, createKernel, run_QS_simulations
from VALIDATION_METRICS import compute_rmse, compute_spem
from PLOTTING import plot_maps, animate_results, plot_time_metric

#%% -- Load data ---

# inDir = r"X:\LoicGerber\knn_image_generation\syntheticImageGeneration\data\voltaData\voltaClean"
inDir = '/home/lgerber8/scratch_idyst/LoicGerber/knn_image_generation/syntheticImageGeneration/data/voltaData/voltaClean'
# outDir = r"X:\LoicGerber\knn_image_generation"
outDir = '/home/lgerber8/scratch_idyst/LoicGerber/knn_image_generation'

et   = xr.open_dataset(os.path.join(inDir, "et.nc"))
pre  = xr.open_dataset(os.path.join(inDir, "pre.nc"))
tmax = xr.open_dataset(os.path.join(inDir, "tmax.nc"))
tavg = xr.open_dataset(os.path.join(inDir, "tavg.nc"))
tmin = xr.open_dataset(os.path.join(inDir, "tmin.nc"))

lulc_map = tf.imread(os.path.join(inDir, "lulc_0p25.tif"))

t_start = "1980-01-01"
t_end   = "2020-12-31"

var_weights = [0.5, 0.5, 2.0, 1.0] # either weights for each variable, or weights per variable per time lag (then length must be time_window * n_variables)
doy_weight  = 1.0
k_neighbors = 25
time_window = 4

et   =   et.sel(time=slice(t_start, t_end))
pre  =  pre.sel(time=slice(t_start, t_end))
tmax = tmax.sel(time=slice(t_start, t_end))
tavg = tavg.sel(time=slice(t_start, t_end))
tmin = tmin.sel(time=slice(t_start, t_end))

pre  =  pre.sel(time=et.time)
tmax = tmax.sel(time=et.time)
tavg = tavg.sel(time=et.time)
tmin = tmin.sel(time=et.time)

# use the same names for spatial coordinates
pre  =  pre.rename({'longitude': 'lon', 'latitude': 'lat'})
tmax = tmax.rename({'longitude': 'lon', 'latitude': 'lat'})
tavg = tavg.rename({'longitude': 'lon', 'latitude': 'lat'})
tmin = tmin.rename({'longitude': 'lon', 'latitude': 'lat'})

# normalise predictors
pre_norm  = normalize_predictors(pre, "precip")
tmax_norm = normalize_predictors(tmax, "temp")
tavg_norm = normalize_predictors(tavg, "temp")
tmin_norm = normalize_predictors(tmin, "temp")

#%% -- Validation with best hyperparameters ---

X_train, y_train, X_query, y_query, dates_train, dates_query = prepare_knn_data(
    et_ds       = et,
    pre_ds      = pre_norm,
    tmax_ds     = tmax_norm,
    tmin_ds     = tmin_norm,
    time_window = time_window,
    mode        = "validation",
    periods     = [
        ("2000-01-01", "2000-02-28"),
        # ("1980-01-01", "1980-12-31"),
        # ("1985-01-01", "1985-12-31"),
        # ("1990-01-01", "1990-12-31"),
        # ("1995-01-01", "1995-12-31"),
        # ("2000-01-01", "2000-12-31"),
        # ("2005-01-01", "2005-12-31"),
        # ("2010-01-01", "2010-12-31"),
        # ("2015-01-01", "2015-12-31"),
        # ("2020-01-01", "2020-12-31")
        ]
    )

kNN_distances, kNN_indices = knn_analogue_selection(
    X_train     = X_train,
    X_query     = X_query,
    dates_train = dates_train,
    dates_query = dates_query,
    k_neighbors = k_neighbors,
    doy_weight  = doy_weight,
    var_weights = var_weights
)

#%% --- Weighted mean approach ---

y_test_mean, y_test_std = knn_image_generation(
    distances  = kNN_distances,
    indices    = kNN_indices,
    y_train    = y_train
)

target_var = list(et.data_vars)[0]
et_da      = et[target_var]
lat        = et_da["lat"].values
lon        = et_da["lon"].values

indices = [0, 50]#, 100] # <---------------------------------- CHOOSE DAYS TO VISUALISE

plot_maps(
    y_obs   = y_query,
    y_sim   = y_test_mean,
    y_std   = y_test_std,
    dates   = dates_query,
    lon     = lon,
    lat     = lat,
    indices = indices
)

rmse_test, segments = compute_rmse(y_query, y_test_mean, dates_query)
spem_test, _        = compute_spem(y_query, y_test_mean, dates_query)

plot_time_metric(dates_query, rmse_test, segments, metric_name="RMSE", ylabel="RMSE [mm/day]")
plot_time_metric(dates_query, spem_test, segments, metric_name="SPEM", ylabel="SPEM [-]")

print(f"Overall RMSE: {np.nanmean(rmse_test):.3f}")
print(f"Overall SPEM: {np.nanmean(spem_test):.3f}")

#%% -- MPS approach ---

# Kernel parameters
ki = createKernel(
    layer_sizes     = [3, 3],           # size of kernel (height, width)
    layer_values    = [3, 1, 1, 1, 1],  # value for each layer: 0: target, 1: lat, 2: lon, 3: lulc, 4: kNN  distance
    map_type        = [2, 2, 2, 2, 3],  # 0: uniform, 1: gaussian, 2: exponential, 3: NaN except central pixel
    sigma_gaus      = 50,               # sigma for gaussian maps
    expo_scale      = 0.02              # scaling factor for exponential maps
    )

# QS parameters
g2s_params = (
    # '-sa', 'mercury.gaia.unil.ch', # server address
    # '-sa', 'localhost',             # use local server
    '-sa', 'tesla-k20c.gaia.unil.ch', # alternative server
    '-a',  'qs',                      # algorithm
    '-dt', [0, 0, 0, 1, 0],           # data type, 0: continuous, 1: categorical
    '-k',   1.2,                      # number of candidates, accepts floats
    '-n',  [10, 5, 5, 5, 5],          # number of neighbours 
    '-j',   0.5,                      # computing power
    # '-s',   42                       # random seed
)

# Tester autoQS?

di, lat_map, lon_map = prepare_TIs_DI(y_train, et, lulc_map=lulc_map)

qs_sims, qs_stds = run_QS_simulations(
    y_train            = y_train,
    kNN_indices        = kNN_indices,
    kNN_distances      = kNN_distances,
    di                 = di,
    ki                 = ki,
    g2s_params         = g2s_params,
    lat_map            = lat_map,
    lon_map            = lon_map,
    lulc_map           = lulc_map,
    n_realizations     = 1,
    mask               = True,  # use if NaNs in target
    mask_seed          = None,  # None for random
    use_distance_layer = True
)

target_var = list(et.data_vars)[0]
et_da      = et[target_var]
lat        = et_da["lat"].values
lon        = et_da["lon"].values

indices = [0, 50]#, 100] # <---------------------------------- CHOOSE DAYS TO VISUALISE

plot_maps(
    y_obs   = y_query,
    y_sim   = qs_sims[0], # first realization
    y_std   = None,#qs_stds,
    dates   = dates_query,
    lon     = lon,
    lat     = lat,
    indices = indices
)

rmse_test, segments = compute_rmse(y_query, qs_sims[0], dates_query)
spem_test, _        = compute_spem(y_query, qs_sims[0], dates_query)

plot_time_metric(dates_query, rmse_test, segments, metric_name="RMSE", ylabel="RMSE [mm/day]")
plot_time_metric(dates_query, spem_test, segments, metric_name="SPEM", ylabel="SPEM [-]")

print(f"Overall RMSE with MPS: {np.nanmean(rmse_test):.3f}")
print(f"Overall SPEM with MPS: {np.nanmean(spem_test):.3f}")

filename = os.path.join(outDir, "qs_simulation.gif") # .gif or .mp4

animate_results(
    y_true   = y_query,
    y_pred   = qs_sims[0],   # choose realization
    dates    = dates_query,
    lon      = lon,
    lat      = lat,
    filename = filename,
    show_uncertainty = None, #qs_stds,
    fps      = 6
)

