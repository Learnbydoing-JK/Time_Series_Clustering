import data_processing as dp
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm

start_time = time.time()
np.random.seed(0)

'''get data from multiple csv files and merge them into one pandas dataframe'''
path = r'data'
data = dp.get_multiple_csvs(path)
data.fillna(method='ffill')
comp_symbols = data.Name.unique()

'''convert DataFrame columns into numpy arrays'''
adj_close_series = dp.col_as_ts(data, comp_symbols, col='Adj Close', normalization='zscore')
distance_matrix = dp.get_distance_matrix(adj_close_series)

'''KMeans clustering: Evaluated with Silhouette Analysis'''
n_clusters_range = [11,12,13]

for n in n_clusters_range:
    plt.figure()
    plt.xlim([-0.1, 1])
    plt.ylim([0, len(adj_close_series)+(n+1)*10])

    clusterer = KMeans(n_clusters=n, random_state=10, n_init=50, n_jobs=-1, precompute_distances=False)
    kmeans = clusterer.fit_predict(adj_close_series)
    silhouette_avg = metrics.silhouette_score(adj_close_series, kmeans)
    print("For n_clusters =", n, "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = metrics.silhouette_samples(adj_close_series, kmeans)

    y_lower = 10

    for i in range(n):
        ith_cluster_silhouette_values = sample_silhouette_values[kmeans == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


plt.show()

clusterer = KMeans(n_clusters=8, random_state=10, n_init=50, n_jobs=-1, precompute_distances=True)
kmeans = clusterer.fit_predict(distance_matrix)
