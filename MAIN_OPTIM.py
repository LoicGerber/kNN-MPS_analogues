#%%
import tifffile as tf
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from itertools import product
from skopt import gp_minimize
from skopt.space import Real
import os

from kNN_FUNC import prepare_knn_data, normalize_predictors, knn_analogue_selection, knn_image_generation
from DIMS_REDUCTION import apply_pca, rf_feature_selection, flatten_X
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

#%% -- Hyperparameter tuning ---

k_list = [10, 15, 25, 40]
time_windows = [0, 2, 4, 7]

results = []

for k, tw in product(k_list, time_windows):

    print(f"Testing k={k}, time_window={tw}")

    X_train, y_train, X_query, y_query, dates_train, dates_query = prepare_knn_data(
        et_ds       = et,
        pre_ds      = pre,
        tmax_ds     = tmax,
        tmin_ds     = tmin,
        time_window = tw,
        mode        = "validation",
        periods     = [
            ("1980-01-01", "1980-12-31"),
            ("1985-01-01", "1985-12-31"),
            ("1990-01-01", "1990-12-31"),
            ("1995-01-01", "1995-12-31"),
            ("2000-01-01", "2000-12-31"),
            ("2005-01-01", "2005-12-31"),
            ("2010-01-01", "2010-12-31"),
            ("2015-01-01", "2015-12-31"),
            ("2020-01-01", "2020-12-31")
        ]
    )

    kNN_distances, kNN_indices = knn_analogue_selection(
        X_train     = X_train,
        X_query     = X_query,
        dates_train = dates_train,
        dates_query = dates_query,
        k_neighbors = k,
        doy_weight  = 0.0,
        var_weights = [1.0, 1.0, 1.0] # neutral weights
    )

    y_mean, _ = knn_image_generation(
        distances  = kNN_distances,
        indices    = kNN_indices,
        y_train    = y_train
    )

    rmse, _ = compute_rmse(y_query, y_mean, dates_query)
    mean_rmse = np.nanmean(rmse)

    results.append((k, tw, mean_rmse))

# --- Select best ---
results = sorted(results, key=lambda x: x[2])
best_k, best_tw, best_rmse = results[0]

print(f"\nBest structure: k={best_k}, time_window={best_tw}")
print(f"RMSE = {best_rmse:.3f}")

#%% -- Hyperparameter optimization ---

X_train, y_train, X_query, y_query, dates_train, dates_query = prepare_knn_data(
        et_ds       = et,
        pre_ds      = pre,
        tmax_ds     = tmax,
        tmin_ds     = tmin,
        time_window = best_tw,
        mode        = "validation",
        periods     = [
            ("1980-01-01", "1980-12-31"),
            ("1985-01-01", "1985-12-31"),
            ("1990-01-01", "1990-12-31"),
            ("1995-01-01", "1995-12-31"),
            ("2000-01-01", "2000-12-31"),
            ("2005-01-01", "2005-12-31"),
            ("2010-01-01", "2010-12-31"),
            ("2015-01-01", "2015-12-31"),
            ("2020-01-01", "2020-12-31")
        ]
    )

search_space = [
    Real(0.1, 5.0, name="w_pre"),
    Real(0.1, 5.0, name="w_tmax"),
    Real(0.1, 5.0, name="w_tmin"),
    Real(0.0, 2.0, name="doy_weight"),
]

def objective(params):
    w_pre, w_tmax, w_tmin, doy_weight = params

    # --- normalize weights ---
    var_weights = np.array([w_pre, w_tmax, w_tmin])
    var_weights /= var_weights.mean()

    kNN_distances, kNN_indices = knn_analogue_selection(
        X_train      = X_train,
        X_query      = X_query,
        dates_train  = dates_train,
        dates_query  = dates_query,
        k_neighbors  = best_k,
        doy_weight   = doy_weight,
        var_weights  = var_weights
    )

    y_mean, _ = knn_image_generation(
        distances  = kNN_distances,
        indices    = kNN_indices,
        y_train    = y_train
    )

    rmse, _ = compute_rmse(y_query, y_mean, dates_query)
    return np.nanmean(rmse)

res = gp_minimize(
    objective,                  # Objective function to minimize
    search_space,               # Hyperparameter search space
    n_calls          = 10,      # Number of evaluations of the objective function
    n_initial_points = 10,      # Number of initial random evaluations before using the surrogate model
    acq_func         = "EI",    # Expected Improvement
    random_state     = 42       # Random seed
)

w_pre, w_tmax, w_tmin, doy_weight = res.x

var_weights = np.array([w_pre, w_tmax, w_tmin])
var_weights /= var_weights.mean()

print("\nOptimal metric parameters")
print(f"Variable weights: pre = {var_weights[0]:.2f}, "
      f"tmax = {var_weights[1]:.2f}, "
      f"tmin = {var_weights[2]:.2f}")
print(f"DOY weight: {doy_weight:.2f}")
print(f"Validation RMSE: {res.fun:.3f}")

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

indices = [0, 50, 100] # <---------------------------------- CHOOSE DAYS TO VISUALISE

plot_maps(
    y_obs   = y_query,
    y_mean  = y_test_mean,
    y_std   = y_test_std,
    dates   = dates_query,
    lon     = lon,
    lat     = lat,
    indices = indices
)

rmse_test, segments = compute_rmse(y_query, y_test_mean, dates_query)

for i, seg in enumerate(segments, 1):
    year = dates_query[seg][0].astype('datetime64[Y]').astype(int) + 1970
    mean_rmse = np.nanmean(rmse_test[seg])
    plt.figure(figsize=(10,4))
    plt.plot(dates_query[seg], rmse_test[seg], '-')
    plt.axhline(mean_rmse, color='red', linestyle='--', label=f"Mean RMSE = {mean_rmse:.2f}")
    plt.ylim(0)
    plt.xlim(dates_query[seg][0], dates_query[seg][-1])
    plt.ylabel("RMSE [mm/day]")
    plt.xlabel("Date")
    plt.legend()
    plt.title(f"RMSE - {year}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print(f"Overall RMSE: {np.nanmean(rmse_test):.3f}")