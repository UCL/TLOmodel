"""Collection of shared fixtures"""
import pytest

DEFAULT_SEED = 83563095832589325021


def pytest_addoption(parser):
    parser.addoption(
        "--seed",
        type=int,
        nargs="*",
        default=[DEFAULT_SEED],
        help="Seed(s) for simulation-level random number generator in tests"
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


def pytest_generate_tests(metafunc):
    if "seed" in metafunc.fixturenames:
        metafunc.parametrize("seed", metafunc.config.getoption("seed"), scope="session")
