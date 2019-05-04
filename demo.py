import data_processing as dp
import time
import numpy as np

if __name__ == '__main__':
    start_time = time.time()
    np.random.seed(0)

    '''get data from multiple csv files and merge them into one pandas dataframe'''
    print("------------------------Retrieving Data------------------------\n")
    path = r'data'
    data = dp.get_multiple_csvs(path)
    data.fillna(method='ffill')
    comp_symbols = data.Name.unique()
    print(data)
    print("-----------Data retrieved, Preprocessing Data...-------------\n")

    '''convert DataFrame columns into numpy arrays'''
    adj_close_series = dp.col_as_ts(data, comp_symbols, col='Adj Close', normalization=None)
    print("---------------Data Processed, Clustering...-----------------\n")


    hclust_dtw = [5,6,3,0,0,5,6,6,0,5,0,0,7,5,0,5,5,7,0,5,0,5,0,0,6,0,0,0,3,5,5,0,3,3,6,6,2,6,3,6,5,0,0,6,0,
                     0,0,5,5,3,3,0,0,0,3,2,5,5, 5, 3, 5, 5, 3, 5, 1, 5, 1, 5, 0, 5, 0, 6, 5, 0, 0, 0, 0, 6, 0,
                     5, 5, 6, 0, 5, 5, 0, 7, 0, 3, 6, 0, 0, 5, 6, 4, 6, 5, 5, 5, 0, 5, 3, 0, 3, 5, 5, 5, 5, 6,
                     5, 0, 0, 0, 0, 6, 0, 5, 0, 0, 0, 5, 0, 5, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 5, 0, 5, 0, 0,
                     5, 0, 5, 5, 5, 2, 2, 5, 0,5, 5, 0, 5, 0, 5, 7, 5, 3, 5, 5, 5, 5, 0, 5, 3, 0, 5, 1, 5, 5,
                     3, 5, 5, 6, 0, 5]

    hclust = [2, 0, 0, 6, 6, 2, 0, 0, 6, 2, 6, 6, 1, 2, 6, 2, 2, 1, 2, 2, 6, 2, 6, 6, 0, 6, 6, 6, 0, 2, 2, 6, 0,
              0, 0, 0, 5, 0, 0, 0, 2, 6, 6, 0, 6, 6, 6, 2, 2, 0, 0, 6, 6, 6, 0, 7, 2, 2, 2, 0, 2, 2, 0, 2, 1, 2,
              1, 2, 2, 2, 6, 0, 2, 6, 6, 6, 2, 0, 6, 2, 2, 0, 6, 2, 2, 6, 1, 6, 0, 0, 6, 6, 2, 0, 4, 0, 2, 2, 2,
              6, 2, 0, 6, 0, 2, 2, 2, 2, 0, 2, 6, 6, 6, 0, 0, 6, 2, 6, 6, 6, 2, 6, 2, 2, 6, 6, 6, 6, 2, 6, 6, 6,
              6, 2, 2, 6, 2, 6, 6, 2, 6, 2, 2, 2, 3, 3, 2, 6, 2, 2, 6, 2, 0, 2, 1, 2, 0, 2, 2, 2, 2, 6, 2, 0, 6,
              2, 1, 2, 2, 0, 2, 2, 0, 2, 2]

    kmeans_zscore = [6, 3, 3, 0, 0, 6, 3, 3, 0, 6, 0, 0, 2, 6, 0, 6, 6, 2, 0, 6, 0, 6, 0, 0, 3, 0, 0, 0, 3, 6, 6,
                     0, 3, 3, 3, 3, 1, 3, 3, 3, 6, 0, 0, 3, 0, 0, 0, 6, 6, 3, 3, 0, 0, 0, 3, 5, 6, 6, 6, 3, 6, 6,
                     3, 6, 2, 6, 2, 6, 0, 6, 6, 3, 6, 0, 0, 0, 0, 3, 0, 6, 6, 3, 0, 6, 6, 0, 2, 0, 3, 3, 0, 0, 6,
                     0, 7, 3, 6, 6, 6, 0, 6, 3, 0, 3, 6, 6, 6, 6, 3, 6, 0, 0, 0, 0, 3, 0, 6, 0, 0, 0, 6, 0, 6, 6,
                     0, 0, 0, 0, 6, 0, 0, 0, 6, 6, 6, 0, 6, 0, 0, 6, 0, 6, 6, 6, 4, 4, 6, 0, 6, 6, 0, 6, 0, 6, 2,
                     6, 3, 6, 6, 6, 6, 0, 6, 3, 0, 6, 2, 6, 6, 3, 6, 6, 3, 0, 6]

    kmeans_dtw = [5, 4, 2, 1, 1, 5, 4, 4, 1, 5, 4, 1, 7, 5, 1, 5, 5, 7, 1, 5, 1, 1, 1, 1, 4, 1, 1, 1, 2, 5, 5, 1,
                   2, 2, 4, 4, 0, 4, 2, 4, 5, 1, 1, 4, 1, 1, 4, 5, 5, 2, 2, 1, 1, 1, 2, 0, 5, 5, 5, 2, 5, 5, 2, 5,
                   3, 5, 3, 5, 1, 5, 5, 4, 5, 1, 4, 1, 1, 4, 1, 5, 5, 4, 1, 5, 5, 1, 7, 1, 2, 4, 1, 1, 5, 4, 6, 4,
                   5, 5, 5, 1, 5, 2, 1, 2, 5, 5, 5, 5, 4, 5, 1, 1, 1, 4, 4, 1, 5, 4, 1, 1, 5, 1, 5, 5, 1, 1, 1, 1,
                   5, 1, 1, 1, 5, 5, 5, 1, 5, 1, 1, 5, 1, 1, 5, 5, 0, 0, 5, 1, 5, 5, 1, 5, 4, 5, 7, 5, 2, 1, 5, 1,
                   5, 1, 5, 2, 4, 5, 3, 5, 5, 2, 5, 5, 4, 1, 5]

    print("------------------Clusters formed, Calculating mean returns----------------\n")

    mean_all = np.mean(adj_close_series, axis=0)
    initial = 100000
    index_invest = 100000
    avg_stock_hold = initial / mean_all[0]
    avg_stock_gain = avg_stock_hold * mean_all[-1] #186481.2748423422

    print("------------------------Index Fund Strategy------------------------\n")
    print("Initial Capital: 100000 USD")
    print("3 Years Investment: 186481.27 USD (tax not included)")
    print("3 Years Return: {}%\n".format((avg_stock_gain/initial)*100))

    print("------------------------Hierarchical DTW Fund Strategy------------------------\n")
    pl_dtw = dp.get_profit(hclust_dtw, comp_symbols, data)
    for i in range(len(pl_dtw)):
        print("------------------------Cluster {}------------------------".format(i+1))
        print("Initial Capital: 100000 USD")
        print("3 Years Investment: {} USD (tax not included)".format(round(pl_dtw[i])))
        print("3 Years Return: {}%".format((pl_dtw[i]/initial)*100))
    print('\n')

    print("------------------------Hierarchical Ward's Method Fund Strategy------------------------\n")
    pl_h = dp.get_profit(hclust, comp_symbols, data)
    for i in range(len(pl_h)):
        print("------------------------Cluster {}------------------------".format(i+1))
        print("Initial Capital: 100000 USD")
        print("3 Years Investment: {} USD (tax not included)".format(round(pl_h[i])))
        print("3 Years Return: {}%".format((pl_h[i]/initial)*100))
    print('\n')

    print("------------------------K-Means Fund Strategy------------------------\n")
    pl_k_z = dp.get_profit(kmeans_zscore, comp_symbols, data)
    for i in range(len(pl_k_z)):
        print("------------------------Cluster {}------------------------".format(i+1))
        print("Initial Capital: 100000 USD")
        print("3 Years Investment: {} USD (tax not included)".format(round(pl_k_z[i])))
        print("3 Years Return: {}%".format((pl_k_z[i]/initial)*100))
    print("\n")

    print("------------------------K-Means DTW Fund Strategy------------------------\n")
    pl_k_z = dp.get_profit(kmeans_dtw, comp_symbols, data)
    for i in range(len(pl_k_z)):
        print("------------------------Cluster {}------------------------".format(i+1))
        print("Initial Capital: 100000 USD")
        print("3 Years Investment: {} USD (tax not included)".format(round(pl_k_z[i])))
        print("3 Years Return: {}%".format((pl_k_z[i]/initial)*100))
    print("--- process done in %s seconds ---" % (time.time() - start_time))