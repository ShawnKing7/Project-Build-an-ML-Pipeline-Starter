import pytest
import pandas as pd

def pytest_addoption(parser):
    parser.addoption("--csv", action="store", default="clean_sample.csv",
                   help="Path to cleaned data CSV")
    parser.addoption("--ref", action="store", default="sample1.csv",
                   help="Path to reference data CSV")
    parser.addoption("--kl-threshold", type=float, default=0.2,
                   help="KL divergence threshold")
    parser.addoption("--min-price", type=float, default=10,
                   help="Minimum price threshold")
    parser.addoption("--max-price", type=float, default=350,
                   help="Maximum price threshold")

@pytest.fixture(scope="session")
def data(request):
    """Load test dataset from local CSV"""
    return pd.read_csv(request.config.getoption("--csv"))

@pytest.fixture(scope="session")
def ref_data(request):
    """Load reference dataset from local CSV"""
    return pd.read_csv(request.config.getoption("--ref"))

# Keep these fixtures unchanged
@pytest.fixture(scope="session")
def kl_threshold(request):
    return request.config.getoption("--kl-threshold")

@pytest.fixture(scope="session")
def min_price(request):
    return request.config.getoption("--min-price")

@pytest.fixture(scope="session")
def max_price(request):
    return request.config.getoption("--max-price")