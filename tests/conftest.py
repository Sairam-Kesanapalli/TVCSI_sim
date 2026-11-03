import pytest

from tvcsi.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def estimator():
    """The frozen S4 estimator of the default configuration (loaded from results/estimators/, or
    fitted there deterministically on the first run, about 1.5 minutes)."""
    from tvcsi.estimator import get_estimator
    return get_estimator(load_config())
