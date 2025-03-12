import pandas as pd

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df
