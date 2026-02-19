import numpy as np
import psutil
from tqdm import tqdm
import xarray as xr
from sklearn.neighbors import NearestNeighbors


def normalize_predictors(da, ptype):
    """
    Normalize predictors for kNN.

    Parameters
    ----------
    da : xarray.DataArray
        Predictor DataArrays (time, lat, lon)
    ptype : str
        Each entry must be one of:
        - "precip"
        - "temp"

    Returns
    -------
    norm_da : xarray.DataArray
        Normalized predictor
    """
    
    if ptype == "precip":
        # log(x + 1), safe for zeros
        da = np.log1p(da)
        # robust scaling using Q5–Q95
        q5  = da.quantile(0.05, dim="time", skipna=True)
        q95 = da.quantile(0.95, dim="time", skipna=True)

        da_norm = (da - q5) / (q95 - q5)
        da_norm = da_norm.clip(0, 1)

    elif ptype == "temp":
        # robust scaling using Q5–Q95
        q5  = da.quantile(0.05, dim="time", skipna=True)
        q95 = da.quantile(0.95, dim="time", skipna=True)

        da_norm = (da - q5) / (q95 - q5)
        da_norm = da_norm.clip(0, 1)

    else:
        raise ValueError(f"Unknown predictor type: {ptype}")

    return da_norm

def prepare_knn_data(
    et_ds,
    time_window,
    mode="validation",
    periods=None,
    pre_ds=None,
    tmax_ds=None,
    tavg_ds=None,
    tmin_ds=None,
):
    """
    Unified data preparation for validation and production kNN ET reconstruction.
    """

    # --- Target ---
    target_var = list(et_ds.data_vars)[0]
    et_da = et_ds[target_var]
    time_dim = "time"

    # --- Predictors ---
    predictors = []
    for ds in (pre_ds, tmax_ds, tavg_ds, tmin_ds):
        if ds is not None:
            predictors.append(ds[list(ds.data_vars)[0]])

    if len(predictors) == 0:
        raise ValueError("At least one predictor must be provided.")

    # --- Feature builder ---
    def build_features(predictor_das):
        feat_list = []
        for da in predictor_das:
            feat_list.append(da)
            for lag in range(1, time_window):
                feat_list.append(da.shift({time_dim: lag}))
        features = xr.concat(feat_list, dim="feature")
        return features.isel({time_dim: slice(time_window, None)})
    
    # =========================
    # VALIDATION MODE
    # =========================
    if mode == "validation":
        
        time = et_da[time_dim]

        if periods is None:
            raise ValueError("periods must be provided for validation.")

        test_mask = np.zeros(time.size, dtype=bool)
        for start, end in periods:
            test_mask |= (
                (time >= np.datetime64(start)) &
                (time <= np.datetime64(end))
            )

        train_mask = ~test_mask

        predictors = [da.sel({time_dim: time}) for da in predictors]

        features = build_features(predictors)
        et_eff   = et_da.isel({time_dim: slice(time_window, None)})

        train_mask = train_mask[time_window:]
        test_mask  = test_mask[time_window:]

        X_train_da = features.isel(time=train_mask)
        X_test_da  = features.isel(time=test_mask)
        y_train_da = et_eff.isel(time=train_mask)
        y_test_da  = et_eff.isel(time=test_mask)

        spatial_dims = [d for d in et_eff.dims if d != time_dim]

        return (
            X_train_da.transpose(time_dim, "feature", *spatial_dims).values,
            y_train_da.transpose(time_dim, *spatial_dims).values,
            X_test_da.transpose(time_dim, "feature", *spatial_dims).values,
            y_test_da.transpose(time_dim, *spatial_dims).values,
            y_train_da[time_dim].values,
            y_test_da[time_dim].values,
        )

    # =========================
    # PRODUCTION MODE
    # =========================
    elif mode == "production":
        
        time = predictors[0][time_dim]

        if periods is None:
            raise ValueError("periods must be provided for production mode.")

        prod_mask = np.zeros(time.size, dtype=bool)
        for start, end in periods:
            prod_mask |= (
                (time >= np.datetime64(start)) &
                (time <= np.datetime64(end))
            )

        train_mask = ~prod_mask

        predictors = [da.sel({time_dim: time}) for da in predictors]

        features = build_features(predictors)
        et_eff   = et_da.isel({time_dim: slice(time_window, None)})

        prod_mask  = prod_mask[time_window:]
        train_mask = train_mask[time_window:]

        X_train_da = features.isel(time=train_mask)
        X_prod_da  = features.isel(time=prod_mask)
        train_mask_et = train_mask[: et_eff.sizes[time_dim]]
        y_train_da = et_eff.isel(time=train_mask_et)

        spatial_dims = [d for d in et_eff.dims if d != time_dim]

        return (
            X_train_da.transpose(time_dim, "feature", *spatial_dims).values,
            y_train_da.transpose(time_dim, *spatial_dims).values,
            X_prod_da.transpose(time_dim, "feature", *spatial_dims).values,
            y_train_da[time_dim].values,
            X_prod_da[time_dim].values,
        )

    else:
        raise ValueError("mode must be 'validation' or 'production'.")

def doy_distance(doy_train, doy_query):
    """
    Cyclic DOY distance (range 0–182).
    """
    diff = np.abs(doy_train - doy_query)
    return np.minimum(diff, 365 - diff)

def knn_analogue_selection(X_train, X_query, dates_train, dates_query, k_neighbors, doy_weight=0.0, var_weights=None):
    """
    KNN analogue prediction with uncertainty from analogue spread.

    Parameters
    ----------
    X_train : (T_train, n_features, latX, lonX)
    y_train : (T_train, latY, lonY)
    X_query : (T_query, n_features, latX, lonX)
    k_neighbors : int

    Returns
    -------
    y_mean : (T_query, latY, lonY)
    y_std  : (T_query, latY, lonY)
    """

    T_train, n_features, n_latX, n_lonX = X_train.shape
    T_query = X_query.shape[0]

    NX = n_latX * n_lonX

    # ---- X masking ----
    X_all = np.concatenate([X_train, X_query], axis=0)
    valid_X = np.all(np.isfinite(X_all), axis=(0, 1))
    mask_X = valid_X.ravel()

    Xtr = X_train.reshape(T_train, n_features, NX)[:, :, mask_X]
    Xq  = X_query.reshape(T_query, n_features, NX)[:, :, mask_X]
    
     # ---- variable weighting ----
    if var_weights is not None:
        var_weights = np.asarray(var_weights)
        
        if var_weights.size != n_features:
            n_vars = var_weights.size
            assert n_features % n_vars == 0

            n_lags = n_features // n_vars

            # expand weights over lags
            feature_weights = np.repeat(var_weights, n_lags)
            
        else:
            assert var_weights.size == n_features
            
            feature_weights = var_weights

        scale = np.sqrt(feature_weights)[:, None]

        Xtr = Xtr * scale
        Xq  = Xq  * scale

    Xtr_vec = Xtr.reshape(T_train, -1)
    Xq_vec  =  Xq.reshape(T_query, -1)

    # ---- Neighbour search ----
    nn = NearestNeighbors(
        n_neighbors=k_neighbors,
        metric="euclidean",
        n_jobs=-1
    )
    nn.fit(Xtr_vec)

    k_search = min(k_neighbors * 5, T_train)
    distances, indices = nn.kneighbors(Xq_vec, n_neighbors=k_search)
    
    if doy_weight > 0:
        doy_train = np.array(dates_train).astype("datetime64[D]").astype(object)
        doy_train = np.array([d.timetuple().tm_yday for d in doy_train])
        
        doy_query = np.array(dates_query).astype("datetime64[D]").astype(object)
        doy_query = np.array([d.timetuple().tm_yday for d in doy_query])
        
        doy_dist = np.zeros_like(distances)

        for i in range(indices.shape[0]):
            doy_dist[i] = doy_distance(
                doy_train[indices[i]],
                doy_query[i]
            )

        # normalize DOY distance to [0, 1]
        doy_dist /= 182.5
        
        distances = distances + doy_weight * doy_dist

    # --- RE-RANK neighbours ---
    order     = np.argsort(distances, axis=1)
    indices   = np.take_along_axis(indices, order, axis=1)[:, :k_neighbors]
    distances = np.take_along_axis(distances, order, axis=1)[:, :k_neighbors]
    
    return distances, indices


def knn_image_generation(distances, indices, y_train):
    """
    Generate ET images from selected analogues.

    Returns
    -------
    y_mean : (Tq, lat, lon)
    y_std  : (Tq, lat, lon)
    """

    Tq, k = indices.shape
    T_train, n_lat, n_lon = y_train.shape
    NY = n_lat * n_lon

    # ---- Y masking ----
    valid_Y = np.all(np.isfinite(y_train), axis=0)
    mask_Y = valid_Y.ravel()

    ytr = y_train.reshape(T_train, NY)[:, mask_Y]

    # ---- weights ----
    weights = 1.0 / (distances + 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)

    # ---- analogue stack ----
    y_pred = np.empty((Tq, k, ytr.shape[1]))
    for i in range(Tq):
        y_pred[i] = ytr[indices[i]]

    # ---- mean & uncertainty ----
    y_mean_flat = np.sum(weights[:, :, None] * y_pred, axis=1)
    y_var_flat  = np.sum(
        weights[:, :, None] * (y_pred - y_mean_flat[:, None, :])**2,
        axis=1
    )

    y_std_flat = np.sqrt(y_var_flat)

    # ---- restore spatial shape ----
    y_mean = np.full((Tq, NY), np.nan)
    y_std  = np.full((Tq, NY), np.nan)

    y_mean[:, mask_Y] = y_mean_flat
    y_std[:,  mask_Y] = y_std_flat

    return (
        y_mean.reshape(Tq, n_lat, n_lon),
        y_std.reshape(Tq, n_lat, n_lon),
    )






def knn_pixelwise_selection(
    X_train, X_query,
    dates_train, dates_query,
    k_neighbors,
    doy_weight=0.0,
    var_weights=None,
    memory_frac=0.6,
    show_progress=True
):
    """
    Production-grade pixelwise kNN analogue search.

    Automatically batches queries to avoid RAM overflow.

    Parameters
    ----------
    memory_frac : float
        Fraction of available RAM allowed for computation.
    """

    T_train, n_feat, ny, nx = X_train.shape
    Tq = X_query.shape[0]
    n_pixels = ny*nx

    # --------------------------------------------------
    # reshape → pixels become batch dimension
    # --------------------------------------------------
    Xtr = X_train.transpose(0,2,3,1).reshape(T_train, n_pixels, n_feat)
    Xq  = X_query.transpose(0,2,3,1).reshape(Tq, n_pixels, n_feat)

    # --------------------------------------------------
    # variable weights
    # --------------------------------------------------
    if var_weights is not None:
        var_weights = np.asarray(var_weights)

        if var_weights.size != n_feat:
            n_vars = var_weights.size
            n_lags = n_feat // n_vars
            feature_weights = np.repeat(var_weights, n_lags)
        else:
            feature_weights = var_weights

        scale = np.sqrt(feature_weights)
        Xtr *= scale
        Xq  *= scale

    # --------------------------------------------------
    # DOY preparation
    # --------------------------------------------------
    if doy_weight > 0:

        doy_train = np.array(dates_train).astype("datetime64[D]").astype(object)
        doy_train = np.array([d.timetuple().tm_yday for d in doy_train])

        doy_query = np.array(dates_query).astype("datetime64[D]").astype(object)
        doy_query = np.array([d.timetuple().tm_yday for d in doy_query])

    # --------------------------------------------------
    # allocate outputs
    # --------------------------------------------------
    indices   = np.empty((Tq, n_pixels, k_neighbors), dtype=np.int32)
    distances = np.empty((Tq, n_pixels, k_neighbors), dtype=np.float32)

    # --------------------------------------------------
    # automatic batch size estimation
    # --------------------------------------------------
    available_mem = psutil.virtual_memory().available * memory_frac

    bytes_per_float = 8
    est_bytes_per_query = T_train * n_pixels * bytes_per_float

    batch_size = max(1, int(available_mem // est_bytes_per_query))

    if show_progress:
        print(f"Auto batch size: {batch_size}")

    # --------------------------------------------------
    # main batch loop
    # --------------------------------------------------
    iterator = range(0, Tq, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Analogue search")

    for q0 in iterator:

        q1 = min(q0 + batch_size, Tq)
        Xqb = Xq[q0:q1]                         # (B, pixels, feat)

        # distance computation
        diff = Xqb[:,None,:,:] - Xtr[None,:,:,:]
        dist = np.sqrt(np.sum(diff**2, axis=-1))   # (B, T_train, pixels)

        # DOY penalty
        if doy_weight > 0:
            doy_diff = np.abs(doy_query[q0:q1,None] - doy_train[None,:])
            doy_diff = np.minimum(doy_diff, 365-doy_diff)/182.5
            dist += doy_weight * doy_diff[:,:,None]

        # select k smallest
        idx = np.argpartition(dist, k_neighbors, axis=1)[:,:k_neighbors,:]
        dist_k = np.take_along_axis(dist, idx, axis=1)

        # sort neighbors
        order = np.argsort(dist_k, axis=1)
        idx   = np.take_along_axis(idx, order, axis=1)
        dist_k= np.take_along_axis(dist_k, order, axis=1)

        indices[q0:q1]   = idx.transpose(0,2,1)
        distances[q0:q1] = dist_k.transpose(0,2,1)

    # reshape back to maps
    indices   = indices.reshape(Tq, ny, nx, k_neighbors)
    distances = distances.reshape(Tq, ny, nx, k_neighbors)

    return distances, indices

def knn_pixelwise_prediction(distances, indices, y_train):
    """
    Build prediction maps from pixel-wise analogues.
    """

    Tq, ny, nx, k = indices.shape
    y_mean = np.full((Tq, ny, nx), np.nan)
    y_std  = np.full((Tq, ny, nx), np.nan)

    for q in range(Tq):
        for iy in range(ny):
            for ix in range(nx):

                ind = indices[q, iy, ix]
                dist = distances[q, iy, ix]

                if np.any(ind < 0):
                    continue

                vals = y_train[ind, iy, ix]

                w = 1/(dist + 1e-12)
                w /= w.sum()

                y_mean[q, iy, ix] = np.sum(w * vals)
                y_std[q, iy, ix]  = np.sqrt(np.sum(w*(vals - y_mean[q,iy,ix])**2))

    return y_mean, y_std
