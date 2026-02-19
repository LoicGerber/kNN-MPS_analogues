#%%
import tifffile as tf
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

from kNN_FUNC import knn_pixelwise_prediction, knn_pixelwise_selection, prepare_knn_data, normalize_predictors, knn_analogue_selection, knn_image_generation
from QS_FUNC import prepare_TIs_DI, createKernel, run_QS_simulations
from VALIDATION_METRICS import compute_rmse
from PLOTTING import plot_maps, animate_results

#%% -- Load data ---

dir = r"X:\LoicGerber\knn_image_generation\syntheticImageGeneration\data\voltaData\voltaClean"

et   = xr.open_dataset(os.path.join(dir, "et.nc"))
pre  = xr.open_dataset(os.path.join(dir, "pre.nc"))
tmax = xr.open_dataset(os.path.join(dir, "tmax.nc"))
tavg = xr.open_dataset(os.path.join(dir, "tavg.nc"))
tmin = xr.open_dataset(os.path.join(dir, "tmin.nc"))

lulc = tf.imread(os.path.join(dir, "lulc_0p25.tif"))

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

kNN_distances, kNN_indices = knn_pixelwise_selection(
    X_train       = X_train,
    X_query       = X_query,
    dates_train   = dates_train,
    dates_query   = dates_query,
    k_neighbors   = k_neighbors,
    doy_weight    = doy_weight,
    var_weights   = var_weights,
    memory_frac   = 0.5,
    show_progress = True
)

y_mean, y_std = knn_pixelwise_prediction(
    kNN_distances = kNN_distances,
    kNN_indices   = kNN_indices,
    y_train       = y_train
)
