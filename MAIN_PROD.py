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

#%% -- Production with best hyperparameters ---

X_train, y_train, X_query, dates_train, dates_prod = prepare_knn_data(
    et_ds       = et,
    pre_ds      = pre_norm,
    tmax_ds     = tmax_norm,
    tavg_ds     = tavg_norm,
    tmin_ds     = tmin_norm,
    time_window = 4,    # <------ NUMBER OF PRIOR DAYS TO INCLUDE
    mode        = "production",
    periods     = [
        ("1965-01-01", "1979-12-31"),
        ("2021-01-01", "2021-12-31")
        ]
    )

print("Training:", dates_train[0], "-", dates_train[-1])
print("Production:", dates_prod[0], "-", dates_prod[-1])

y_prod_mean, y_prod_std = knn_analogue_selection(
    X_train     = X_train,
    y_train     = y_train,
    X_query     = X_query,
    k_neighbors = 20,
    doy_weight  = 0.0,
    var_weights = [1.0, 1.0, 1.0, 1.0]
)

print("Production ET generated for:")
print(dates_prod[0], "-", dates_prod[-1])

target_var = list(et.data_vars)[0]
et_da      = et[target_var]
lat        = et_da["lat"].values
lon        = et_da["lon"].values

plot_maps(
    y_obs   = None,
    y_mean  = y_prod_mean,
    y_std   = y_prod_std,
    dates   = dates_prod,
    lon     = lon,
    lat     = lat,
    indices = [0, 120, 365] # <------------------------------------ CHANGE DAYS TO VISUALIZE HERE (based on index)
)

# Spatial mean uncertainty per day
uncertainty_time_mean = np.nanmean(y_prod_std, axis=(1, 2))

plt.figure(figsize=(10, 3))
plt.plot(dates_prod, uncertainty_time_mean, lw=1.5)
plt.ylabel("Mean uncertainty [mm/day]")
plt.xlabel("Date")
plt.title("Spatially averaged ET uncertainty")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Mean uncertainty map
mean_uncertainty_map = np.nanmean(y_prod_std, axis=0)
plt.figure(figsize=(5, 4))
im = plt.pcolormesh(lon, lat, mean_uncertainty_map, shading="auto", cmap="inferno")
plt.gca().set_aspect("equal")
plt.title("Mean ET uncertainty")
plt.xticks([]); plt.yticks([])
c = plt.colorbar(im, fraction=0.046, pad=0.04)
c.set_label("Mean uncertainty [mm/day]")
plt.tight_layout()
plt.show()

# Histogram of uncertainty
plt.figure(figsize=(5, 3))
plt.hist(y_prod_std.ravel(), bins=50, density=True, alpha=0.8)
plt.xlabel("Uncertainty [mm/day]")
plt.ylabel("Density")
plt.title("Distribution of ET uncertainty")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Save NetCDF
ds_prod = xr.Dataset(
    data_vars=dict(
        et_mean=(("time", "lat", "lon"), y_prod_mean),
        et_uncertainty=(("time", "lat", "lon"), y_prod_std),
    ),
    coords=dict(time=dates_prod, lat=lat, lon=lon),
    attrs=dict(
        title="Analogue-based evapotranspiration production run",
        method="kNN climate analogue reconstruction",
        uncertainty="Analogue spread (weighted standard deviation)",
        period="2021–2025",
        units="mm day-1",
    ),
)
ds_prod["et_mean"].attrs = dict(
    long_name="Daily evapotranspiration (mean reconstruction)",
    units="mm day-1",
    description="Weighted mean of k nearest climate analogues"
)
ds_prod["et_uncertainty"].attrs = dict(
    long_name="Evapotranspiration uncertainty",
    units="mm day-1",
    description="Weighted standard deviation across climate analogues"
)

output_path = "ET_production_2021_2025_mean_uncertainty.nc" # <---------------------- CHANGE FILE NAME HERE
ds_prod.to_netcdf(output_path)
