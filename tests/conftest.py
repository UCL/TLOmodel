"""Collection of shared fixtures"""
import pytest

DEFAULT_SEED = 83563095832589325021


def pytest_addoption(parser):
    parser.addoption(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for simulation random number generator"
    )
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="--skip-slow option is set")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture
def seed(request):
    return request.config.getoption("--seed")
