"""Forward modeling of single-crystal diffuse scattering."""

from discordant_parsimony.structure import Structure, Sites
from discordant_parsimony.pairs import (
    PairTerms,
    pair_terms_from_covariance,
    thermal_laue_terms,
)
from discordant_parsimony.scattering import evaluate

try:
    from discordant_parsimony._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    "Structure",
    "Sites",
    "PairTerms",
    "pair_terms_from_covariance",
    "thermal_laue_terms",
    "evaluate",
]
