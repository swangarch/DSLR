#!/usr/bin/python3

import pandas as pd
import os


def load(path: str) -> pd.DataFrame:
    """Load a csv file and return it's pandas.DataFrame object."""

    try:
        assert isinstance(path, str), "Wrong file path type."
        assert os.path.exists(path), "Data file not exists."
        assert path.endswith(".csv"), "Not csv data."

        df = pd.read_csv(path, header=0, index_col=0)
        print("Loading dataset of dimensions", df.shape)
        return df

    except AssertionError as e:
        print("AssertionError:", e)
        return None

    except Exception as e:
        print("Error:", e)
        return None
