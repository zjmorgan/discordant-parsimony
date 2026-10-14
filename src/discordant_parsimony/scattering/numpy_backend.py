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


def bin_factor(d, widths, projection=None):
    r"""
    Attenuation of pair terms by averaging over histogram bins.

    A bin spans widths :math:`\Delta u_a` along the axes of the volume,
    whose Miller indices are :math:`\mathbf{h} = W\mathbf{u}`. Since
    :math:`\mathbf{h}\cdot\mathbf{d} = \mathbf{u}\cdot(W^\mathsf{T}
    \mathbf{d})`, averaging :math:`e^{2\pi i \mathbf{h}\cdot\mathbf{d}}`
    over the bin multiplies a term by

    .. math::

        \prod_a \operatorname{sinc}\!\left(\Delta u_a\,
        (W^\mathsf{T}\mathbf{d})_a\right).

    Parameters
    ----------
    d : ndarray, (n, 3)
        Pair vectors in fractional coordinates.
    widths : array_like, (3,)
        Bin widths along the volume axes ``u``.
    projection : ndarray, (3, 3), optional
        The projection matrix :math:`W` of the volume; identity (axes
        along h, k, l) by default.

    Returns
    -------
    factor : ndarray, (n,)

    """

    d = np.asarray(d, dtype=float)
    if projection is not None:
        d = d @ np.asarray(projection, dtype=float)
    return np.prod(np.sinc(np.asarray(widths, dtype=float) * d), axis=1)


def design_matrix(
    h,
    orbits,
    sites,
    structure,
    widths=None,
    thermal=True,
    b=None,
    max_elements=2**22,
    operators=None,
    projection=None,
):
    r"""
    Intensity per unit covariance of each pair orbit.

    Column :math:`k` is

    .. math::

        A_k(\mathbf{h}) = \frac{1}{n_W} \sum_{W} \sum_{p \in k} b_i b_j\,
        e^{-W_i - W_j}
        \cos(2\pi\, \mathbf{h} \cdot R_W\mathbf{d}_p)\,
        \beta(R_W \mathbf{d}_p),

    averaged over rotations :math:`R_W` (``h @ W``), with the bin factor
    :math:`\beta` of :func:`bin_factor` and Debye-Waller factors at
    :math:`\mathbf{h} R_W`, so that :math:`I = A \mathbf{c}` for orbit
    coefficients :math:`\mathbf{c}`. The bin is the measured one around
    :math:`\mathbf{h}` for every image, which is why the rotation acts
    on the pair vector inside the bin factor.

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
        Histogram bin widths along the volume axes for exact bin
        averaging.
    thermal : bool, optional
        Include Debye-Waller factors.
    b : array_like, (n,), optional
        Scattering lengths in fm. Defaults to neutron values.
    max_elements : int, optional
        Tile size bound on Q-by-pair products.
    operators : ndarray of int, (n, 3, 3), optional
        Average over ``h @ W`` for these rotations, e.g. the coset
        representatives of a data symmetry higher than that of the
        structure
        (:func:`~discordant_parsimony.selection.coset_operators`).
    projection : ndarray, (3, 3), optional
        Projection matrix of the volume (``Volume.W``) when its axes are
        not along h, k, l.

    Returns
    -------
    A : ndarray, (m, n_orbits)

    """

    from scipy.sparse import csr_matrix

    h = np.asarray(h, dtype=float).reshape(-1, 3)
    b = sites.b if b is None else np.asarray(b)
    if operators is None or len(operators) == 0:
        operators = [np.eye(3)]
    U6 = _tensor_components(sites.U)
    tile = max(1, max_elements // max(1, len(orbits)))

    A = np.zeros((len(h), orbits.n_orbits))
    for W in operators:
        W = np.asarray(W, dtype=float)
        # (h W) . d = h . (W d): rotate the pair vectors
        d = orbits.d @ W.T
        w = b[orbits.i] * b[orbits.j]
        if widths is not None:
            w = w * bin_factor(d, widths, projection)
        S = csr_matrix(
            (w, (np.arange(len(orbits)), orbits.orbit)),
            shape=(len(orbits), orbits.n_orbits),
        )
        for start in range(0, len(h), tile):
            hs = h[start : start + tile]
            X = np.cos(2 * np.pi * hs @ d.T)
            if thermal:
                Q6 = _quadratic_components(structure.Q(hs @ W))
                dw = np.exp(-0.5 * Q6 @ U6.T)
                X *= dw[:, orbits.i] * dw[:, orbits.j]
            A[start : start + tile] += (S.T @ X.T).T
    return A / len(operators)


def size_effect_design_matrix(
    h,
    orbits,
    vectors,
    sites,
    structure,
    widths=None,
    thermal=True,
    b=None,
    max_elements=2**22,
    operators=None,
    projection=None,
):
    r"""
    Intensity per unit size-effect coefficient.

    To first order in the displacement :math:`\mathbf{v} = n\mathbf{u}`
    of present atoms, the occupation-displacement cross term is

    .. math::

        I_\mathrm{size}(\mathbf{Q}) = -\sum_p b_i b_j\,
        (\mathbf{Q} \cdot \boldsymbol{\sigma}_p)
        \sin(2\pi\, \mathbf{h} \cdot \mathbf{d}_p),
        \qquad
        \boldsymbol{\sigma}_p = \langle \delta n_i(\mathbf{0})\,
        \mathbf{v}_j(\mathbf{R}) \rangle
        - \langle \mathbf{v}_i(\mathbf{0})\,
        \delta n_j(\mathbf{R}) \rangle,

    with :math:`\boldsymbol{\sigma}_p = \sum_k \theta_k
    \mathbf{v}_{pk}` from :func:`~discordant_parsimony.symmetry.vector_orbits`.
    It is odd in :math:`\mathbf{Q} - \mathbf{G}` about Bragg positions
    (asymmetric diffuse) and grows linearly with :math:`|\mathbf{Q}|`.
    Debye-Waller, bin and symmetry averaging follow
    :func:`design_matrix`.

    Parameters
    ----------
    h : array_like, (m, 3)
        Miller indices in r.l.u.
    orbits : PairOrbits
    vectors : VectorOrbits
    sites : Sites
    structure : Structure
    widths, thermal, b, max_elements, operators, projection
        As for :func:`design_matrix`.

    Returns
    -------
    A : ndarray, (m, len(vectors))
        Columns in units of intensity per Angstrom of coefficient.

    """

    from scipy.sparse import csr_matrix

    from discordant_parsimony.symmetry import vector_slots

    h = np.asarray(h, dtype=float).reshape(-1, 3)
    b = sites.b if b is None else np.asarray(b)
    if operators is None or len(operators) == 0:
        operators = [np.eye(3)]
    index, value = vector_slots(vectors, len(orbits))
    U6 = _tensor_components(sites.U)
    tile = max(1, max_elements // max(1, 3 * len(orbits)))

    A = np.zeros((len(h), len(vectors)))
    for W in operators:
        W = np.asarray(W, dtype=float)
        d_rot = orbits.d @ W.T  # (h W) . d = h . (W d)
        w = b[orbits.i] * b[orbits.j]
        if widths is not None:
            w = w * bin_factor(d_rot, widths, projection)
        for start in range(0, len(h), tile):
            hs = h[start : start + tile]
            hw = hs @ W
            X = -np.sin(2 * np.pi * hw @ orbits.d.T) * w
            if thermal:
                Q6 = _quadratic_components(structure.Q(hw))
                dw = np.exp(-0.5 * Q6 @ U6.T)
                X *= dw[:, orbits.i] * dw[:, orbits.j]
            Q = structure.Q(hw)
            for s in range(index.shape[1]):
                k = index[:, s]
                used = k >= 0
                proj = Q @ value[used, s].T  # (t, pairs)
                S = csr_matrix(
                    (np.ones(used.sum()), (np.arange(used.sum()), k[used])),
                    shape=(used.sum(), len(vectors)),
                )
                A[start : start + tile] += (S.T @ (X[:, used] * proj).T).T
    return A / len(operators)


def displacement_design_matrix(
    h,
    orbits,
    tensors,
    sites,
    structure,
    widths=None,
    thermal=True,
    b=None,
    max_elements=2**22,
    operators=None,
    projection=None,
):
    r"""
    Intensity per unit displacement-displacement coefficient.

    To second order in the displacement :math:`\mathbf{v} = n\mathbf{u}`
    of present atoms, correlated displacements contribute

    .. math::

        I_\mathrm{disp}(\mathbf{Q}) = \sum_p b_i b_j\,
        (\mathbf{Q}^{\mathsf{T}} V_p \mathbf{Q})
        \cos(2\pi\, \mathbf{h} \cdot \mathbf{d}_p)\, e^{-W_i - W_j},
        \qquad
        V_p = \langle \mathbf{v}_i(\mathbf{0})\,
        \mathbf{v}_j(\mathbf{R})^{\mathsf{T}} \rangle,

    with :math:`V_p = \sum_k \theta_k V_{pk}` from
    :func:`~discordant_parsimony.symmetry.tensor_orbits`. It grows as
    :math:`|\mathbf{Q}|^2`. Debye-Waller, bin and symmetry averaging
    follow :func:`design_matrix`.

    Parameters
    ----------
    h : array_like, (m, 3)
    orbits : PairOrbits
    tensors : TensorOrbits
    sites : Sites
    structure : Structure
    widths, thermal, b, max_elements, operators, projection
        As for :func:`design_matrix`.

    Returns
    -------
    A : ndarray, (m, len(tensors))
        Columns in units of intensity per Angstrom^2 of coefficient.

    """

    from scipy.sparse import csr_matrix

    from discordant_parsimony.symmetry import coefficient_slots

    h = np.asarray(h, dtype=float).reshape(-1, 3)
    b = sites.b if b is None else np.asarray(b)
    if operators is None or len(operators) == 0:
        operators = [np.eye(3)]
    index, value = coefficient_slots(tensors, len(orbits), (3, 3))
    U6 = _tensor_components(sites.U)
    tile = max(1, max_elements // max(1, 6 * len(orbits)))

    A = np.zeros((len(h), len(tensors)))
    for W in operators:
        W = np.asarray(W, dtype=float)
        d_rot = orbits.d @ W.T
        w = b[orbits.i] * b[orbits.j]
        if widths is not None:
            w = w * bin_factor(d_rot, widths, projection)
        for start in range(0, len(h), tile):
            hw = h[start : start + tile] @ W
            X = np.cos(2 * np.pi * hw @ orbits.d.T) * w
            Q6 = _quadratic_components(structure.Q(hw))
            if thermal:
                dw = np.exp(-0.5 * Q6 @ U6.T)
                X *= dw[:, orbits.i] * dw[:, orbits.j]
            for s in range(index.shape[1]):
                k = index[:, s]
                used = k >= 0
                proj = Q6 @ _tensor_components(value[used, s]).T
                S = csr_matrix(
                    (np.ones(used.sum()), (np.arange(used.sum()), k[used])),
                    shape=(used.sum(), len(tensors)),
                )
                A[start : start + tile] += (S.T @ (X[:, used] * proj).T).T
    return A / len(operators)
