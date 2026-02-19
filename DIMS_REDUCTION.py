import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

def flatten_X(X):
    T, F, ny, nx = X.shape
    return X.reshape(T, F*ny*nx)

def apply_pca(X_train, X_query, variance_ratio=0.99):
    """
    Flatten X_train and X_query, keep only valid pixels (non-NaN), and apply PCA.

    Parameters
    ----------
    X_train : (T, n_vars, ny, nx)
    X_query : (Tq, n_vars, ny, nx)
    variance_ratio : float, PCA explained variance threshold

    Returns
    -------
    Xtr_p : PCA-transformed training data (T, n_components)
    Xq_p  : PCA-transformed query data (Tq, n_components)
    pca    : fitted PCA object
    valid_mask : boolean mask of kept pixels (ny*nx)
    """
    T, n_vars, ny, nx = X_train.shape
    Tq = X_query.shape[0]
    NY = ny * nx

    # --- Mask pixels with any NaNs across time in training ---
    valid_mask = np.all(np.isfinite(X_train), axis=0).all(axis=0)  # shape (ny, nx)
    
    # flatten X and keep only valid pixels
    X_train_flat = X_train.reshape(T, n_vars, NY)[:, :, valid_mask.ravel()].reshape(T, -1)
    X_query_flat = X_query.reshape(Tq, n_vars, NY)[:, :, valid_mask.ravel()].reshape(Tq, -1)

    # --- Apply PCA ---
    pca = PCA(n_components=variance_ratio, svd_solver="full")
    Xtr_p = pca.fit_transform(X_train_flat)
    Xq_p  = pca.transform(X_query_flat)

    print(f"PCA reduced features: {X_train_flat.shape[1]} → {Xtr_p.shape[1]}")
    
    return Xtr_p, Xq_p, pca, valid_mask

def rf_feature_selection(X_train, y_train, n_keep=200):
    """
    Random Forest feature importance for spatio-temporal predictors.

    Parameters
    ----------
    X_train : (T, n_vars, ny, nx)
    y_train : (T, ny, nx)
    
    Returns
    -------
    idx : top feature indices
    importance : full importance array
    """
    T, n_vars, ny, nx = X_train.shape
    
    # valid pixels (no NaNs in y)
    valid_mask = np.all(np.isfinite(y_train), axis=0)
    
    # flatten X keeping only valid pixels
    Xtr = X_train.reshape(T, n_vars*ny*nx)[:, np.repeat(valid_mask.ravel(), n_vars)]
    ytr = np.nanmean(y_train[:, valid_mask], axis=1)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
    rf.fit(Xtr, ytr)
    
    importance = rf.feature_importances_
    idx = np.argsort(importance)[::-1][:n_keep]
    
    return idx, importance
