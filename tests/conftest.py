from pathlib import Path

import pytest

from discordant_parsimony import Structure
from discordant_parsimony.io import data_root

FIXTURES = Path(__file__).parent / "fixtures"
TB5RU6SN18 = data_root() / "Tb5Ru6Sn18"

requires_tb5ru6sn18 = pytest.mark.skipif(
    not (TB5RU6SN18 / "Tb5Ru6Sn18.cif").exists(),
    reason="private Tb5Ru6Sn18 data not available (set DIFFUSE_DATA_ROOT)",
)


@pytest.fixture
def tetragonal():
    return Structure.from_cif(FIXTURES / "split_tetragonal.cif")


@pytest.fixture
def tb5ru6sn18():
    return Structure.from_cif(TB5RU6SN18 / "Tb5Ru6Sn18.cif")
