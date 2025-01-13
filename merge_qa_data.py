# code to merge all previously deduplicated qa datas into a single dataset
import pandas as pd
import os

# load all deduplicated datasets
def load_deduplicated_data(path):
    return pd.read_csv(path)
