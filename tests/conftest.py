import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synthetic import make_dataset  # noqa: E402


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("data") / "dataset", seed=0, n_per_country=40)
