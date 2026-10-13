r"""Average crystal structure: lattice, symmetry, and atomic sites.

Conventions
-----------
Fractional coordinates :math:`\mathbf{x}` and Miller indices
:math:`\mathbf{h}` (r.l.u.) refer to the cell given in the CIF. Direct
basis vectors are the columns of :math:`A` (Cartesian,
:math:`\mathring{\mathrm{A}}`), with :math:`\mathbf{a}` along Cartesian
:math:`x` and :math:`\mathbf{b}` in the :math:`xy` plane. Reciprocal
basis vectors (without the :math:`2\pi`) are the columns of
:math:`B = (A^{-1})^\mathsf{T}`, so that

.. math::

    \mathbf{Q} = 2\pi B \mathbf{h}, \qquad
    \mathbf{Q} \cdot (A \mathbf{x}) = 2\pi\, \mathbf{h} \cdot \mathbf{x},

with :math:`\mathbf{Q}` in :math:`\mathring{\mathrm{A}}^{-1}`.
Displacement parameters :math:`U` are stored as Cartesian covariance
tensors in :math:`\mathring{\mathrm{A}}^2`.
"""

import re
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from discordant_parsimony.tables import scattering_length


def _value(text):
    """Parse a CIF number, dropping any standard uncertainty."""

    return float(re.sub(r"\(.*\)", "", text))


def parse_symop(text):
    """
    Parse a symmetry operator such as ``"-x+1/2, y, z+1/2"``.

    Parameters
    ----------
    text : str
        Coordinate triplet in CIF ``xyz`` notation.

    Returns
    -------
    W : ndarray, (3, 3)
        Rotation part acting on fractional coordinates.
    t : ndarray, (3,)
        Translation part in fractional coordinates.

    """

    W, t = np.zeros((3, 3)), np.zeros(3)
    components = text.replace(" ", "").lower().split(",")
    if len(components) != 3:
        raise ValueError(f"Invalid symmetry operator {text!r}")
    for i, component in enumerate(components):
        for term in re.findall(r"[+-]?[^+-]+", component):
            sign = -1 if term[0] == "-" else 1
            body = term.lstrip("+-")
            if body[-1] in "xyz":
                coefficient = Fraction(body[:-1].rstrip("*") or 1)
                W[i, "xyz".index(body[-1])] += sign * coefficient
            else:
                t[i] += sign * Fraction(body)
    return W, t


def lattice_vectors(a, b, c, alpha, beta, gamma):
    """
    Direct lattice vectors as matrix columns.

    Parameters
    ----------
    a, b, c : float
        Lattice constants in Angstrom.
    alpha, beta, gamma : float
        Lattice angles in degrees.

    Returns
    -------
    A : ndarray, (3, 3)
        Columns are the Cartesian direct basis vectors.

    """

    alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])
    cx = np.cos(beta)
    cy = (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(1 - cx**2 - cy**2)
    return np.array(
        [
            [a, b * np.cos(gamma), c * cx],
            [0, b * np.sin(gamma), c * cy],
            [0, 0, c * cz],
        ]
    )


@dataclass
class Sites:
    r"""
    Atomic positions in one unit cell.

    Attributes
    ----------
    label : ndarray of str, (n,)
        Asymmetric-unit label of each position.
    species : ndarray of str, (n,)
        Element or isotope symbol.
    occupancy : ndarray, (n,)
        Average occupancy.
    x : ndarray, (n, 3)
        Fractional coordinates in [0, 1).
    U : ndarray, (n, 3, 3)
        Cartesian displacement covariance in :math:`\mathring{\mathrm{A}}^2`.

    """

    label: np.ndarray
    species: np.ndarray
    occupancy: np.ndarray
    x: np.ndarray
    U: np.ndarray

    def __len__(self):
        return len(self.label)

    def select(self, labels):
        """
        Positions belonging to the given asymmetric-unit labels.

        Parameters
        ----------
        labels : sequence of str
            Labels to keep, e.g. the partially occupied sites.

        Returns
        -------
        sites : Sites
            Subset preserving the original order.

        """

        missing = set(labels) - set(self.label)
        if missing:
            raise KeyError(f"Unknown site labels {sorted(missing)}")
        mask = np.isin(self.label, labels)
        return Sites(
            self.label[mask],
            self.species[mask],
            self.occupancy[mask],
            self.x[mask],
            self.U[mask],
        )

    def partial(self, tol=1e-6):
        """Labels of sites with occupancy below one."""

        labels = self.label[self.occupancy < 1 - tol]
        return list(dict.fromkeys(labels))

    @property
    def b(self):
        """Neutron scattering lengths in fm (real part; must be real)."""

        b = np.array([scattering_length(s) for s in self.species])
        if np.any(b.imag != 0):
            raise NotImplementedError("Complex scattering lengths")
        return b.real


class Structure:
    """
    Average crystal structure from a CIF.

    Parameters
    ----------
    cell : sequence of float
        Lattice parameters ``(a, b, c, alpha, beta, gamma)``.
    operators : list of tuple
        Symmetry operators ``(W, t)`` on fractional coordinates.
    asymmetric : list of dict
        Asymmetric-unit sites with keys ``label``, ``species``,
        ``occupancy``, ``x`` (fractional) and ``U`` (Cartesian).

    """

    def __init__(self, cell, operators, asymmetric):

        self.cell = tuple(float(v) for v in cell)
        self.A = lattice_vectors(*self.cell)
        self.B = np.linalg.inv(self.A).T
        self.operators = operators
        self.asymmetric = asymmetric

    @classmethod
    def from_cif(cls, filename, block=None):
        """
        Read the average structure from a CIF file.

        Parameters
        ----------
        filename : str
            Path to the CIF.
        block : str, optional
            Data block name. The first block is used by default.

        Returns
        -------
        structure : Structure

        """

        import CifFile

        cif = CifFile.ReadCif(str(filename))
        data = cif[block or cif.keys()[0]]

        keys = ("a", "b", "c", "alpha", "beta", "gamma")
        names = [f"_cell_length_{k}" for k in keys[:3]]
        names += [f"_cell_angle_{k}" for k in keys[3:]]
        cell = [_value(data[name]) for name in names]

        for key in (
            "_space_group_symop_operation_xyz",
            "_symmetry_equiv_pos_as_xyz",
        ):
            if key in data:
                operators = [parse_symop(op) for op in data[key]]
                break
        else:
            raise KeyError("No symmetry operators in CIF")

        A = lattice_vectors(*cell)
        N = np.diag(np.linalg.norm(np.linalg.inv(A).T, axis=0))

        aniso = {}
        if "_atom_site_aniso_label" in data:
            components = ("11", "22", "33", "23", "13", "12")
            columns = [data[f"_atom_site_aniso_U_{c}"] for c in components]
            for i, label in enumerate(data["_atom_site_aniso_label"]):
                U11, U22, U33, U23, U13, U12 = (_value(c[i]) for c in columns)
                Ustar = np.array(
                    [[U11, U12, U13], [U12, U22, U23], [U13, U23, U33]]
                )
                aniso[label] = N @ Ustar @ N

        labels = data["_atom_site_label"]
        types = data.get("_atom_site_type_symbol", labels)
        occupancies = data.get("_atom_site_occupancy", ["1"] * len(labels))
        uiso = data.get("_atom_site_U_iso_or_equiv", ["0"] * len(labels))

        asymmetric = []
        for i, label in enumerate(labels):
            x = [_value(data[f"_atom_site_fract_{c}"][i]) for c in "xyz"]
            if label in aniso:
                V = aniso[label]
            else:
                V = _value(uiso[i]) * np.linalg.inv(A) @ np.linalg.inv(A).T
            asymmetric.append(
                {
                    "label": label,
                    "species": re.match(r"\d*[A-Z][a-z]?", types[i]).group(),
                    "occupancy": _value(occupancies[i]),
                    "x": np.array(x),
                    "V": V,
                }
            )

        structure = cls(cell, operators, [])
        for site in asymmetric:
            V = site.pop("V")
            site["U"] = A @ V @ A.T
            structure.asymmetric.append(site)
        return structure

    def sites(self, tol=1e-4):
        """
        Expand the asymmetric unit to all positions in one unit cell.

        Parameters
        ----------
        tol : float, optional
            Fractional distance below which positions are merged.

        Returns
        -------
        sites : Sites

        """

        Ainv = np.linalg.inv(self.A)
        label, species, occupancy, x, U = [], [], [], [], []
        for site in self.asymmetric:
            V = Ainv @ site["U"] @ Ainv.T
            unique = []
            for W, t in self.operators:
                xp = (W @ site["x"] + t) % 1
                xp[np.isclose(xp, 1, atol=tol)] = 0
                if unique:
                    diff = (np.array(unique) - xp + 0.5) % 1 - 0.5
                    if np.min(np.abs(diff).max(axis=1)) < tol:
                        continue
                unique.append(xp)
                label.append(site["label"])
                species.append(site["species"])
                occupancy.append(site["occupancy"])
                x.append(xp)
                U.append(self.A @ W @ V @ W.T @ self.A.T)
        return Sites(
            np.array(label),
            np.array(species),
            np.array(occupancy),
            np.array(x),
            np.array(U),
        )

    def Q(self, h):
        """
        Cartesian scattering vectors.

        Parameters
        ----------
        h : array_like, (..., 3)
            Miller indices in r.l.u.

        Returns
        -------
        Q : ndarray, (..., 3)
            Scattering vectors in inverse Angstrom.

        """

        return 2 * np.pi * np.asarray(h) @ self.B.T

    def distance(self, dx):
        """Length in Angstrom of fractional separation vectors."""

        return np.linalg.norm(np.asarray(dx) @ self.A.T, axis=-1)
