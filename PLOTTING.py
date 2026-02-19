import numpy as np
import matplotlib.pyplot as plt
import imageio
import tempfile
import os

def plot_maps(y_obs, y_sim, y_std, dates, lon, lat, indices):
    """
    Plot observed ET, predicted ET, error, and optionally uncertainty maps.
    """

    # --- Prepare grid ---
    if lon.ndim == 1 and lat.ndim == 1:
        Lon, Lat = np.meshgrid(lon, lat)
    else:
        Lon, Lat = lon, lat

    for idx in indices:
        y_pred = y_sim[idx]
        date   = np.array(dates[idx]).astype("datetime64[D]")

        has_obs = y_obs is not None
        has_unc = y_std is not None

        if has_obs:
            y_ref = y_obs[idx]
            y_err = y_pred - y_ref
            vmin  = np.nanmin([y_ref, y_pred])
            vmax  = np.nanmax([y_ref, y_pred])
            vmax_err = np.nanpercentile(np.abs(y_err), 95)
        else:
            y_ref = None
            y_err = None
            vmin  = np.nanmin(y_pred)
            vmax  = np.nanmax(y_pred)

        if has_unc:
            y_unc = y_std[idx]
            vmax_std = np.nanpercentile(y_unc, 95)

        # --- number of panels ---
        ncols = 1                     # predicted always shown
        if has_obs:
            ncols += 2                # obs + error
        if has_unc:
            ncols += 1                # uncertainty

        fig, axes = plt.subplots(1, ncols, figsize=(4.5*ncols, 4),
                                 constrained_layout=True)

        if ncols == 1:
            axes = [axes]

        col = 0

        # --- Observed ---
        if has_obs:
            im = axes[col].pcolormesh(Lon, Lat, y_ref,
                                      shading="auto", vmin=vmin, vmax=vmax)
            axes[col].set_title(f"Observed ET\n{date}")
            axes[col].set_aspect("equal")
            c = plt.colorbar(im, ax=axes[col])
            c.set_label("ET [mm/day]")
            col += 1

        # --- Predicted ---
        im = axes[col].pcolormesh(Lon, Lat, y_pred,
                                  shading="auto", vmin=vmin, vmax=vmax)
        axes[col].set_title("Predicted ET" if has_obs else f"Predicted ET\n{date}")
        axes[col].set_aspect("equal")
        c = plt.colorbar(im, ax=axes[col])
        c.set_label("ET [mm/day]")
        col += 1

        # --- Error ---
        if has_obs:
            im = axes[col].pcolormesh(Lon, Lat, y_err,
                                      shading="auto",
                                      vmin=-vmax_err, vmax=vmax_err,
                                      cmap="coolwarm")
            axes[col].set_title("Error (Pred − Ref)")
            axes[col].set_aspect("equal")
            c = plt.colorbar(im, ax=axes[col])
            c.set_label("ET [mm/day]")
            col += 1

        # --- Uncertainty ---
        if has_unc:
            im = axes[col].pcolormesh(Lon, Lat, y_unc,
                                      shading="auto",
                                      vmin=0, vmax=vmax_std,
                                      cmap="magma")
            axes[col].set_title("Uncertainty")
            axes[col].set_aspect("equal")
            c = plt.colorbar(im, ax=axes[col])
            c.set_label("ET [mm/day]")

        plt.show()

def plot_time_metric(dates, metric, segments, metric_name="RMSE", ylabel=None):
    """
    Plot a time series metric for each segment with its mean.

    Parameters
    ----------
    dates : np.ndarray
        Array of datetime64 objects for each time step
    metric : np.ndarray
        Metric values (e.g., RMSE or SPEM), same length as dates
    segments : list of np.ndarray
        List of index arrays defining continuous segments
    metric_name : str
        Name of the metric (used in title and legend)
    ylabel : str, optional
        Label for y-axis (defaults to metric_name)
    """
    if ylabel is None:
        ylabel = metric_name

    for i, seg in enumerate(segments, 1):
        year = dates[seg][0].astype('datetime64[Y]').astype(int) + 1970
        mean_val = np.nanmean(metric[seg])

        plt.figure(figsize=(10, 4))
        plt.plot(dates[seg], metric[seg], '-', label=f"{metric_name}")
        plt.axhline(mean_val, color='red', linestyle='--', label=f"Mean = {mean_val:.2f}")
        plt.ylim(0)
        plt.xlim(dates[seg][0], dates[seg][-1])
        plt.ylabel(ylabel)
        plt.xlabel("Date")
        plt.legend()
        plt.title(f"{metric_name} - {year}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def animate_results(
    y_true,
    y_pred,
    dates,
    lon,
    lat,
    filename="animation.gif",
    fps=5,
    dpi=110,
    show_uncertainty=None,  # pass std array or None
):
    """
    High-quality animation of ET simulations vs reference.

    Panels
    ------
    Reference | Simulation | Error | (optional) Uncertainty

    Parameters
    ----------
    y_true : (T, ny, nx)
    y_pred : (T, ny, nx)
    dates  : (T,)
    lon, lat : 1D or 2D arrays
    show_uncertainty : (T,ny,nx) or None
    """

    # ---------- grid ----------
    if lon.ndim == 1 and lat.ndim == 1:
        Lon, Lat = np.meshgrid(lon, lat)
    else:
        Lon, Lat = lon, lat

    T = len(dates)

    # ---------- global color limits ----------
    vmin = np.nanmin([y_true, y_pred])
    vmax = np.nanmax([y_true, y_pred])

    err_all = y_pred - y_true
    vmax_err = np.nanpercentile(np.abs(err_all), 95)

    if show_uncertainty is not None:
        vmax_std = np.nanpercentile(show_uncertainty, 95)

    # ---------- number of panels ----------
    ncols = 4 if show_uncertainty is not None else 3

    frames = []

    with tempfile.TemporaryDirectory() as tmpdir:

        for t in range(T):

            fig, axes = plt.subplots(
                1, ncols,
                figsize=(4.5*ncols, 4),
                constrained_layout=True
            )

            date = np.array(dates[t]).astype("datetime64[D]")

            # ---------- RMSE ----------
            rmse = np.sqrt(np.nanmean((y_pred[t] - y_true[t])**2))

            col = 0

            # ---------- reference ----------
            im0 = axes[col].pcolormesh(
                Lon, Lat, y_true[t],
                shading="auto", vmin=vmin, vmax=vmax
            )
            axes[col].set_title("Reference")
            axes[col].set_aspect("equal")
            c0 = plt.colorbar(im0, ax=axes[col])
            c0.set_label("ET [mm/day]")
            col += 1

            # ---------- prediction ----------
            im1 = axes[col].pcolormesh(
                Lon, Lat, y_pred[t],
                shading="auto", vmin=vmin, vmax=vmax
            )
            axes[col].set_title("Simulation")
            axes[col].set_aspect("equal")
            c1 = plt.colorbar(im1, ax=axes[col])
            c1.set_label("ET [mm/day]")
            col += 1

            # ---------- error ----------
            err = y_pred[t] - y_true[t]
            im2 = axes[col].pcolormesh(
                Lon, Lat, err,
                shading="auto",
                vmin=-vmax_err, vmax=vmax_err,
                cmap="coolwarm"
            )
            axes[col].set_title("Error")
            axes[col].set_aspect("equal")
            c2 = plt.colorbar(im2, ax=axes[col])
            c2.set_label("Error [mm/day]")
            col += 1

            # ---------- uncertainty ----------
            if show_uncertainty is not None:
                im3 = axes[col].pcolormesh(
                    Lon, Lat, show_uncertainty[t],
                    shading="auto",
                    vmin=0, vmax=vmax_std,
                    cmap="magma"
                )
                axes[col].set_title("Uncertainty")
                axes[col].set_aspect("equal")
                c3 = plt.colorbar(im3, ax=axes[col])
                c3.set_label("Std [mm/day]")

            # ---------- title ----------
            fig.suptitle(
                f"{date}   |   RMSE = {rmse:.3f} mm/day",
                fontsize=14
            )

            # ---------- save frame ----------
            path = os.path.join(tmpdir, f"{t:04d}.png")
            fig.savefig(path, dpi=dpi)
            plt.close(fig)

            frames.append(imageio.imread(path))

    # ---------- save ----------
    if filename.endswith(".gif"):
        imageio.mimsave(filename, frames, fps=fps)

    elif filename.endswith(".mp4"):
        imageio.mimsave(filename, frames, fps=fps, codec="libx264")

    else:
        raise ValueError("filename must end with .gif or .mp4")

    print("Saved:", filename)
    
    
    