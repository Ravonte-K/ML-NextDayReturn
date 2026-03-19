import pandas as pd
import numpy as np

def inspectData(path: str):
    df = pd.read_csv(path)
    print(df.isna().sum())
    print(df.duplicated().sum())
    return(df)


if __name__ == "__main__":
    print(inspectData)