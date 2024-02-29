import pandas as pd

def load_data(file_path):
    """
    Load dataset from a CSV file.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)
