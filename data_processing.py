import glob
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.preprocessing import MinMaxScaler
from fastdtw import *



def get_multiple_csvs(path):
    '''
    :param path: path to the folder containing multiple csv files
    :return: pandas dataframe with csv contents stacked
    '''
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

def get_certain_csvs(path, comps):
    '''
    :param path: path to the folder containing multiple csv files
    :return: pandas dataframe with csv contents stacked
    '''
    all_files = glob.glob(path + "/*.csv")

    li = []

    for filename in all_files:
        for name in comps:
            df = pd.read_csv(filename, index_col=None, header=0)
            if len(df.index) != 1259:
                continue
            elif (df.iloc[0]['Name'] == name):
                print(df.iloc[0]['Name'])
                li.append(df)
                continue
            else:
                # li.append(df)
                continue

    frame = pd.concat(li, axis=0, ignore_index=True)

    return frame

def col_as_ts(df, li, col='Adj Close', normalization=None):
    '''

    :param df: Pandas Dataframe
    :param li: List of company names
    :param col: column to extract into numpy arrays
                each row's index correspond to the index
                of the company name in 'li'
    :param normalization: choose a normalization technique for the time series
    :return: a 2D numpy array with rows as time series
    '''
    ts = np.empty((0, 1259))

    for symbols in li:
        curr_row=np.empty((1,1))
        for index, row in df[df['Name'] == symbols].iterrows():
            curr_row = np.insert(curr_row, -1, row[col], axis=1)
        curr_row = np.delete(curr_row, -1, axis=1)
        ts = np.append(ts, curr_row, axis=0)

    if (normalization == 'zscore'):
        ts = sp.stats.zscore(ts)
    elif (normalization == 'minmax'):
        ts = MinMaxScaler().fit_transform(ts)
    elif(normalization == None):
        return ts

    return ts



def get_distance_matrix(array):
    rows = array.shape[0]
    distance_matrix = np.zeros((rows,rows))

    for index in range(0, rows):
        print("current row: %s" % index)
        for i in range(0, rows):
            if i <= index:
                continue
            distance, path = fastdtw(array[index], array[i], dist=sp.spatial.distance.euclidean)
            distance_matrix[index][i] += distance
            distance_matrix[i][index] += distance
    return distance_matrix


def get_clusters(li,comps):
    clusters = [[],[],[],[],[],[],[],[]]
    for i in range(len(li)):
        clusters[li[i]].append(comps[i])
    return clusters


def get_profit(li, comp, df):
    clusters = get_clusters(li, comp)
    profit_list = []
    for cl in clusters:
        acs = col_as_ts(df, cl, col='Adj Close', normalization=None)
        avg = np.mean(acs,axis=0)
        ini = 100000
        hold = ini / avg[0]
        gain = hold * avg[-1]
        profit_list.append(gain)
    return profit_list

