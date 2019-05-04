import data_processing as dp
import time
import numpy as np
from dtaidistance import dtw
import sys

if __name__ == '__main__':
    start_time = time.time()
    np.random.seed(0)

    '''get data from multiple csv files and merge them into one pandas dataframe'''
    path = r'data'
    data = dp.get_multiple_csvs(path)
    data.fillna(method='ffill')
    comp_symbols = data.Name.unique()

    print(data)

    '''convert DataFrame columns into numpy arrays'''
    adj_close_series = dp.col_as_ts(data, comp_symbols, col='Adj Close', normalization='minmax')
    dm = dtw.distance_matrix(adj_close_series)
    print(dm)

    print("--- DTW distance matrix %s seconds ---" % (time.time() - start_time))
    sys.exit()


    print("--- nparray transform in %s seconds ---\n" % (time.time() - start_time))


    linkage_types = ["single", "complete","ward"]
    for lt in linkage_types:
        plt.figure()
        hclust = AgglomerativeClustering(n_clusters=8,
                                        affinity='euclidean',
                                        compute_full_tree=True,
                                        linkage=lt).fit(adj_close_series)
        plt.title('%s Hierarchical Clustering Dendrogram' % lt)
        plot_dendrogram(hclust, labels=comp_symbols)
        plt.show()
    print("--- HIERARCHICAL in %s seconds ---" % (time.time() - start_time))
    print(hclust.labels_)