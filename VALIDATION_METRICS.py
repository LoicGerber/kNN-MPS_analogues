import numpy as np
from sklearn.isotonic import spearmanr

def compute_rmse(y_obs, y_pred, dates, max_gap_days=1):
    """
    Compute RMSE over space for each time step and split by date gaps.

    Returns
    -------
    rmse : (T,)
    segments : list of index arrays for continuous date segments
    """
    T = y_obs.shape[0]
    rmse = np.zeros(T)

    for t in range(T):
        diff = y_obs[t] - y_pred[t]
        rmse[t] = np.sqrt(np.nanmean(diff**2))

    # --- split by temporal gaps ---
    dates  = np.asarray(dates)
    gaps   = np.diff(dates).astype("timedelta64[D]").astype(int)
    breaks = np.where(gaps > max_gap_days)[0]

    segments = []
    start = 0
    for b in breaks:
        segments.append(np.arange(start, b + 1))
        start = b + 1
    segments.append(np.arange(start, T))

    return rmse, segments

def spem2D(x, y):
        # Mask valid entries
        valid = ~(np.isnan(x) | np.isnan(y))
        if not np.any(valid):
            return np.nan

        x_valid = x[valid]
        y_valid = y[valid]

        # Spearman correlation
        try:
            rs, _ = spearmanr(x_valid, y_valid)
            if np.isnan(rs):
                rs = 0.0
        except:
            rs = 0.0

        # Coefficient of variation ratio
        mean_x, mean_y = np.nanmean(x), np.nanmean(y)
        std_x, std_y = np.nanstd(x), np.nanstd(y)
        gamma = 0.0
        if mean_x != 0 and mean_y != 0 and std_x != 0 and std_y != 0:
            gamma = (std_x / mean_x) / (std_y / mean_y)

        # Z-score difference
        alpha = 0.0
        if std_x != 0 and std_y != 0:
            z_diff = ((x - mean_x) / std_x) - ((y - mean_y) / std_y)
            alpha = 1 - np.sqrt(np.nanmean(z_diff**2))

        # SPEM
        spem_val = np.sqrt((rs - 1)**2 + (gamma - 1)**2 + (alpha - 1)**2)
        return spem_val
    
def compute_spem(y_obs, y_pred, dates, max_gap_days=1):
    """
    Compute SPEM over space for each time step and split by date gaps.
    Compatible with compute_rmse.

    Parameters
    ----------
    y_obs : np.ndarray
        Observed values, shape (T, X, Y)
    y_pred : np.ndarray
        Predicted values, same shape as y_obs
    dates : array-like
        Datetimes corresponding to each time step
    max_gap_days : int, optional
        Maximum allowed gap in days for continuous segments

    Returns
    -------
    spem_vals : np.ndarray
        SPEM per time step, shape (T,)
    segments : list of np.ndarray
        List of index arrays for continuous date segments
    """
    
    T = y_obs.shape[0]
    spem_vals = np.zeros(T)

    for t in range(T):
        spem_vals[t] = spem2D(y_obs[t], y_pred[t])

    # --- split by temporal gaps ---
    dates = np.asarray(dates)
    gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
    breaks = np.where(gaps > max_gap_days)[0]

    segments = []
    start = 0
    for b in breaks:
        segments.append(np.arange(start, b + 1))
        start = b + 1
    segments.append(np.arange(start, T))

    return spem_vals, segments

