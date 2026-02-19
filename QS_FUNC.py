import numpy as np
from tqdm import tqdm
from g2s import g2s
from scipy.stats import multivariate_normal

def create_lat_lon_maps(n_lat, n_lon,
                        lat_min, lat_max,
                        lon_min, lon_max,
                        normalize=True):
    """
    Create latitude and longitude raster maps.

    Returns
    -------
    lat_map, lon_map : (n_lat, n_lon)
    """

    lat_vec = np.linspace(lat_max, lat_min, n_lat)
    lon_vec = np.linspace(lon_min, lon_max, n_lon)

    lon_map, lat_map = np.meshgrid(lon_vec, lat_vec)

    if normalize:
        lat_map = (lat_map - lat_map.min()) / (lat_map.max() - lat_map.min())
        lon_map = (lon_map - lon_map.min()) / (lon_map.max() - lon_map.min())

    return lat_map, lon_map

def build_TIs(indices_for_query, y_train,
              lat_map, lon_map, lulc_map):
    """
    Build training images for one query timestep.

    Returns
    -------
    TIs : list of arrays, each (ny, nx, layers)
    """

    if lulc_map is None:
        layers_static = np.stack([lat_map, lon_map], axis=0)
    else:
        layers_static = np.stack([lat_map, lon_map, lulc_map], axis=0)

    TIs = []

    for idx in indices_for_query:

        ti = np.concatenate(
            ([y_train[idx]], layers_static),
            axis=0
        )  # (layers, ny, nx)

        ti = np.moveaxis(ti, 0, -1)  # → (ny, nx, layers)

        TIs.append(ti)

    return TIs

def build_DI(lat_map, lon_map, lulc_map):
    """
    Construct Data Image for QS simulation.
    Target layer is empty (NaN).
    """

    ny, nx = lat_map.shape

    target_empty = np.full((ny, nx), np.nan)

    DI = np.dstack([
        target_empty,
        lat_map,
        lon_map,
        lulc_map
    ])

    return DI

def prepare_TIs_DI(y_train, target, lulc_map=None, normalize=True):
    """
    Build lat/lon maps and base DI for QS.

    Parameters
    ----------
    y_train : (T_train, ny, nx)
    et : xarray.Dataset with lat/lon coordinates
    lulc_map : (ny, nx) or None, default zeros
    normalize : bool

    Returns
    -------
    di : (ny, nx, layers)
    lat_map, lon_map : (ny, nx)
    """
    ny, nx = y_train.shape[1:]
    
    lat_map, lon_map = create_lat_lon_maps(
        n_lat=ny,
        n_lon=nx,
        lat_min=float(target.lat.min()),
        lat_max=float(target.lat.max()),
        lon_min=float(target.lon.min()),
        lon_max=float(target.lon.max()),
        normalize=normalize
    )

    if lulc_map is None:
        di = build_DI(lat_map, lon_map)
    else:
        di = build_DI(lat_map, lon_map, lulc_map)
    
    return di, lat_map, lon_map

def add_distance_layer(TIs, distances_for_query):
    """
    Append normalized analogue-distance layer to each TI.

    Parameters
    ----------
    TIs : list of (ny,nx,layers)
    distances_for_query : (k,)
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    TIs_new : list of (ny,nx,layers+1)
    """

    ny, nx = TIs[0].shape[:2]

    d = np.asarray(distances_for_query, dtype=float)

    # ---- normalize to [0,1] ----
    d_min = np.nanmin(d)
    d_max = np.nanmax(d)
    d_norm = (d - d_min) / (d_max - d_min + 1e-12)

    TIs_new = []

    for ti, dn in zip(TIs, d_norm):
        dist_layer = np.full((ny, nx), dn)
        ti_new = np.dstack([ti, dist_layer])
        TIs_new.append(ti_new)

    return TIs_new

def normalize(arr):
    amin, amax = arr.min(), arr.max()
    return (arr - amin) / (amax - amin) if amax != amin else arr

def generate_map(layer_sizes, mode, param=None):
    h, w = layer_sizes
    if mode == "uniform":
        return np.ones((h, w))
    if mode == "nan":
        arr = np.full((h, w), np.nan)
        arr[h//2, w//2] = 1
        return arr
    # shared grid
    y, x = np.mgrid[:h, :w]
    cy, cx = h // 2, w // 2
    if mode == "gaussian":
        rv = multivariate_normal([cx, cy], param)
        return normalize(rv.pdf(np.dstack((x, y))))
    if mode == "exponential":
        dist = np.sqrt((x-cx)**2 + (y-cy)**2)
        return normalize(np.exp(-dist * param)) if param != 0 else np.ones((h,w))
    raise ValueError("Invalid mode")

def createKernel(layer_sizes, layer_values, map_type,
                 sigma_gaus, expo_scale):
    h, w = layer_sizes
    n = len(layer_values)
    ki = np.empty((h, w, n))
    for i, m in enumerate(map_type):
        if m == 0:
            base = generate_map(layer_sizes, "uniform")
        elif m == 1:
            base = generate_map(layer_sizes, "gaussian", sigma_gaus)
        elif m == 2:
            base = generate_map(layer_sizes, "exponential", expo_scale)
        elif m == 3:
            base = generate_map(layer_sizes, "nan")
        else:
            raise ValueError("Invalid map_type value")
        ki[:, :, i] = base * layer_values[i]
    return ki

def create_simulation_path(y_train, seed=None):
    """
    Create a 2D simulation path: NaNs → -inf, valid pixels → random values.
    
    Uses median across time to detect valid pixels.
    """
    if seed is not None:
        np.random.seed(seed)

    median_map = np.nanmedian(y_train, axis=0)
    valid_mask = np.isfinite(median_map)
    
    sim_path = np.full(median_map.shape, -np.inf, dtype=float)
    sim_path[valid_mask] = np.random.rand(np.sum(valid_mask))
    
    return sim_path

def run_QS_simulations(y_train, kNN_indices, kNN_distances, di, ki, g2s_params,
                       lat_map, lon_map, lulc_map,
                       n_realizations=1, mask=False, mask_seed=None, use_distance_layer=True):
    """
    Run QS for multiple realizations and multiple query times.

    Returns
    -------
    qs_sims : (n_realizations, T_query, ny, nx)
    """
    Tq = len(kNN_indices)
    ny, nx = lat_map.shape

    qs_sims = np.full((n_realizations, Tq, ny, nx), np.nan)
    
    total_iters = n_realizations * Tq
    
    if use_distance_layer:
        ny, nx, _ = di.shape
        di = np.dstack([di, np.zeros((ny, nx))])

    with tqdm(total=total_iters, desc="QS simulations") as pbar:
        for r in range(n_realizations):
            for q in range(Tq):
                # Build TIs (list of ny x nx x layers)
                TIs = build_TIs(
                    kNN_indices[q],
                    y_train,
                    lat_map,
                    lon_map,
                    lulc_map
                )
                
                if use_distance_layer:
                    TIs = add_distance_layer(TIs, kNN_distances[q])
                
                if mask is True:
                    sp = create_simulation_path(y_train, seed=mask_seed)
                else:
                    sp = []

                # Run QS
                sim, *_ = g2s(
                    '-ti', TIs,
                    '-di', di,
                    '-ki', ki,
                    '-sp', sp,
                    *g2s_params
                )

                qs_sims[r, q] = sim[:,:,0]
                
                pbar.set_postfix({"realization": r+1, "query": q+1})
                pbar.update(1)
        
    qs_stds = np.nanstd(qs_sims, axis=0)
    
    return qs_sims, qs_stds

