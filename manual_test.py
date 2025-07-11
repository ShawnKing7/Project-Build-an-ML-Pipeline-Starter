import pandas as pd

df = pd.read_csv("clean_sample_cleaned.csv")

def test_row_count(data: pd.DataFrame):
    assert 1 < data.shape[0] < 1000000, f"Row count {data.shape[0]} is out of range."

def test_price_range(data: pd.DataFrame):
    assert data["price"].between(10, 350).all(), "Out-of-range price detected."

# Run them manually
test_row_count(df)
test_price_range(df)
print("Both tests passed.")
