"""Reading measured volumes and locating private local data.

Experimental data are never stored in the repository. Set the
``DIFFUSE_DATA_ROOT`` environment variable to the directory holding
them; otherwise ``local_data/`` at the repository root (ignored by git)
is used.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def data_root():
    """
    Directory with private experimental data.

    Returns
    -------
    root : pathlib.Path
        ``$DIFFUSE_DATA_ROOT`` if set, else ``<repository>/local_data``.

    """

    if "DIFFUSE_DATA_ROOT" in os.environ:
        return Path(os.environ["DIFFUSE_DATA_ROOT"])
    return Path(__file__).resolve().parents[2] / "local_data"


@dataclass
class Volume:
    r"""
    Histogrammed reciprocal-space volume on a regular grid.

    Attributes
    ----------
    signal : ndarray, (n0, n1, n2)
        Intensity, indexed along axes ``D0, D1, D2``.
    sigma : ndarray, (n0, n1, n2)
        One-standard-deviation uncertainty.
    edges : list of ndarray
        Bin edges along each axis.
    W : ndarray, (3, 3)
        Projection matrix; Miller indices are
        :math:`\mathbf{h} = W \mathbf{u}` for axis coordinates
        :math:`\mathbf{u}`.

    """

    signal: np.ndarray
    sigma: np.ndarray
    edges: list
    W: np.ndarray

    @property
    def centers(self):
        return [0.5 * (e[1:] + e[:-1]) for e in self.edges]

    @property
    def widths(self):
        """Bin widths along each axis (assumed uniform)."""

        return np.array([np.diff(e).mean() for e in self.edges])

    def hkl(self, index):
        """
        Miller indices of bin centres.

        Parameters
        ----------
        index : tuple of ndarray
            Integer bin indices along ``D0, D1, D2``.

        Returns
        -------
        h : ndarray, (..., 3)

        """

        u = np.stack([c[i] for c, i in zip(self.centers, index)], -1)
        return u @ self.W.T

    def index(self, u):
        """Nearest bin indices of axis coordinates ``u`` (..., 3)."""

        u = np.asarray(u, dtype=float)
        return tuple(
            np.clip(np.searchsorted(e, u[..., k]) - 1, 0, len(e) - 2)
            for k, e in enumerate(self.edges)
        )


def load_mdhisto(filename):
    """
    Read a Mantid MDHistoWorkspace saved with ``SaveMD``.

    Parameters
    ----------
    filename : str or pathlib.Path
        NeXus file.

    Returns
    -------
    signal, errors_squared : ndarray
        Arrays indexed along ``D0, D1, D2``.
    edges : list of ndarray
        Bin edges along each axis.
    W : ndarray, (3, 3)
        Projection matrix (identity if not stored).

    """

    import h5py

    with h5py.File(filename, "r") as f:
        g = f["MDHistoWorkspace/data"]
        order = g["signal"].attrs["axes"]
        order = order.decode() if isinstance(order, bytes) else order
        order = order.split(":")
        n = len(order)
        perm = [order.index(f"D{k}") for k in range(n)]
        signal = np.transpose(g["signal"][()], perm)
        errors_squared = np.transpose(g["errors_squared"][()], perm)
        edges = [g[f"D{k}"][()] for k in range(n)]
        key = "MDHistoWorkspace/experiment0/logs/W_MATRIX/value"
        W = f[key][()].reshape(3, 3) if key in f else np.eye(3)
    return signal, errors_squared, edges, W


def reduce_volume(data, norm, background_data, background_norm, scale=1.0):
    """
    Normalised, background-subtracted volume.

    Computes ``data/norm - scale * background_data/background_norm``
    with uncorrelated error propagation. Bins with non-positive
    normalisation are ``nan``.

    Parameters
    ----------
    data, norm, background_data, background_norm : str or pathlib.Path
        MDHisto files of the sample and background measurements.
    scale : float, optional
        Background scale factor.

    Returns
    -------
    volume : Volume

    """

    d, d2, edges, W = load_mdhisto(data)
    n, _, _, _ = load_mdhisto(norm)
    b, b2, _, _ = load_mdhisto(background_data)
    bn, _, _, _ = load_mdhisto(background_norm)

    with np.errstate(divide="ignore", invalid="ignore"):
        signal = d / n - scale * b / bn
        sigma = np.sqrt(d2 / n**2 + scale**2 * b2 / bn**2)
    invalid = (n <= 0) | (bn <= 0)
    signal[invalid] = np.nan
    sigma[invalid] = np.nan
    return Volume(signal, sigma, edges, W)
