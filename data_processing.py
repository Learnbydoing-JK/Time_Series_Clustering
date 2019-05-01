import csv, copy, glob
import pandas as pd

def get_multiple_csvs(path):
    all_files = glob.glob(path + "/*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        if len(df.index) != 1259:
            continue
        else:
            li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    return frame
