"""CPU reference evaluation of pair-term intensities (float64)."""

import numpy as np


def _quadratic_components(Q):
    """Products ``Q_a Q_b`` for the six independent tensor components."""

    Qx, Qy, Qz = Q.T
    return np.stack([Qx * Qx, Qy * Qy, Qz * Qz, Qx * Qy, Qx * Qz, Qy * Qz], 1)


def _tensor_components(U):
    """Matching tensor components, off-diagonals doubled."""

    return np.stack(
        [
            U[:, 0, 0],
            U[:, 1, 1],
            U[:, 2, 2],
            2 * U[:, 0, 1],
            2 * U[:, 0, 2],
            2 * U[:, 1, 2],
        ],
        1,
    )


def evaluate(h, terms, structure, max_elements=2**22, check=False):
    r"""
    Diffuse intensity of pair terms at arbitrary reciprocal positions.

    Computes

    .. math::

        I(\mathbf{Q}) = \sum_p w_p \cos(2\pi\, \mathbf{h} \cdot \mathbf{d}_p)
        \exp\left(-\tfrac{1}{2} \mathbf{Q}^\mathsf{T} U_p \mathbf{Q}\right)

    by direct summation in tiles of at most ``max_elements`` Q-term
    products, never forming the full Q-by-terms matrix.

    Parameters
    ----------
    h : array_like, (..., 3)
        Miller indices in r.l.u.
    terms : PairTerms
        Reversal-closed pair terms.
    structure : Structure
        Supplies the reciprocal metric for the Gaussian factors.
    max_elements : int, optional
        Tile size bound.
    check : bool, optional
        Also accumulate the sine sum and raise if the terms are not
        reversal closed.

    Returns
    -------
    I : ndarray, (...)
        Intensity in :math:`\mathrm{fm}^2` per unit cell.

    """

    h = np.asarray(h, dtype=float)
    shape = h.shape[:-1]
    h = h.reshape(-1, 3)
    d = np.asarray(terms.d, dtype=float)
    w = np.asarray(terms.w, dtype=float)

    U6 = None if terms.U is None else _tensor_components(terms.U)
    tile = max(1, max_elements // max(1, len(w)))

    I = np.empty(len(h))
    imag = 0.0
    for start in range(0, len(h), tile):
        hs = h[start : start + tile]
        phase = 2 * np.pi * hs @ d.T
        weight = np.broadcast_to(w, phase.shape)
        if U6 is not None:
            Q6 = _quadratic_components(structure.Q(hs))
            weight = w * np.exp(-0.5 * Q6 @ U6.T)
        I[start : start + tile] = np.einsum("qp,qp->q", np.cos(phase), weight)
        if check:
            s = np.einsum("qp,qp->q", np.sin(phase), weight)
            imag = max(imag, np.abs(s).max())

    if check:
        scale = max(np.abs(I).max(), np.abs(w).sum() * 1e-12)
        if imag > 1e-8 * scale:
            raise ValueError(f"Terms not reversal closed (|Im| = {imag:g})")
    return I.reshape(shape)


def design_matrix(
    h,
    orbits,
    sites,
    structure,
    widths=None,
    thermal=True,
    b=None,
    max_elements=2**22,
):
    r"""
    Intensity per unit covariance of each pair orbit.

    Column :math:`k` is

    .. math::

        A_k(\mathbf{Q}) = \sum_{p \in k} b_i b_j\, e^{-W_i - W_j}
        \cos(2\pi\, \mathbf{h} \cdot \mathbf{d}_p)
        \prod_a \operatorname{sinc}(\Delta h_a d_{p,a}),

    so that :math:`I = A \mathbf{c}` for orbit coefficients
    :math:`\mathbf{c}`.

    Parameters
    ----------
    h : array_like, (m, 3)
        Miller indices in r.l.u.
    orbits : PairOrbits
        Pairs and their orbits.
    sites : Sites
        Participating positions.
    structure : Structure
        Supplies the reciprocal metric.
    widths : array_like, (3,), optional
        Histogram bin widths in r.l.u. for exact bin averaging.
    thermal : bool, optional
        Include Debye-Waller factors.
    b : array_like, (n,), optional
        Scattering lengths in fm. Defaults to neutron values.
    max_elements : int, optional
        Tile size bound on Q-by-pair products.

    Returns
    -------
    A : ndarray, (m, n_orbits)

    """

    from scipy.sparse import csr_matrix

    h = np.asarray(h, dtype=float).reshape(-1, 3)
    b = sites.b if b is None else np.asarray(b)
    w = b[orbits.i] * b[orbits.j]
    if widths is not None:
        w = w * np.prod(np.sinc(np.asarray(widths) * orbits.d), axis=1)
    S = csr_matrix(
        (w, (np.arange(len(orbits)), orbits.orbit)),
        shape=(len(orbits), orbits.n_orbits),
    )
    U6 = _tensor_components(sites.U)
    tile = max(1, max_elements // max(1, len(orbits)))

    A = np.empty((len(h), orbits.n_orbits))
    for start in range(0, len(h), tile):
        hs = h[start : start + tile]
        X = np.cos(2 * np.pi * hs @ orbits.d.T)
        if thermal:
            dw = np.exp(-0.5 * _quadratic_components(structure.Q(hs)) @ U6.T)
            X *= dw[:, orbits.i] * dw[:, orbits.j]
        A[start : start + tile] = (S.T @ X.T).T
    return A
