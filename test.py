import data_processing as dp
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import silhouette_score, silhouette_samples
import scipy as sp
import csv
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib import style

start_time = time.time()

np.random.seed(0)

path = r'/home/chikunboiwong/Documents/CMPT370/Time_Series_Clustering/sample_data'
data = dp.get_multiple_csvs(path)
comp_symbols = data.Name.unique()

'''convert DataFrame columns into respective dictionaries'''
actual = {}
for symbols in comp_symbols:
    actual[symbols] = {'open':[],
                       'high':[],
                       'low':[],
                       'close':[],
                       'volume':[]}


for index, row in data.iterrows():
    for symbols in comp_symbols:
        if row['Name'] == symbols:
            for key in actual[symbols].keys():
                actual[symbols][key].append(row[key])
        else:
            continue


'''Convert time series in dictionary into numpy arrays (matrices) for scikit learn'''
open_series = np.empty((0,1259))
high_series = np.empty((0,1259))
low_series = np.empty((0,1259))
close_series = np.empty((0,1259))
volume_series = np.empty((0,1259))

for k in actual.keys():
    actual[k]['open'] = np.asarray([sp.stats.zscore(actual[k]['open'])])
    actual[k]['high'] = np.asarray([sp.stats.zscore(actual[k]['high'])])
    actual[k]['low'] = np.asarray([actual[k]['low']])
    actual[k]['close'] = np.asarray([actual[k]['close']])
    actual[k]['volume'] = np.asarray([actual[k]['volume']])


    open_series = np.append(open_series,actual[k]['open'], axis=0)
    high_series = np.append(high_series,actual[k]['high'], axis=0)
    low_series = np.append(low_series,actual[k]['low'], axis=0)
    close_series = np.append(close_series,actual[k]['close'], axis=0)
    volume_series = np.append(volume_series,actual[k]['volume'], axis=0)


print(open_series.shape)

'''KMeans clustering'''
n_clusters_range = [2,3,4,5,6,7,8]

for n in n_clusters_range:
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.set_size_inches(18,7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(open_series)+(n+1)*10])

    clusterer = KMeans(n_clusters=n, random_state=10)
    kmeans = clusterer.fit_predict(open_series)
    silhouette_avg = silhouette_score(open_series, kmeans)
    print("For n_clusters =", n, "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(open_series, kmeans)

    y_lower = 10

    for i in range(n):
        ith_cluster_silhouette_values = sample_silhouette_values[kmeans == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(kmeans.astype(float) / n)
    ax2.scatter(open_series[:, 0], open_series[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n),
                 fontsize=14, fontweight='bold')

plt.show()

'''Kernal PCA to reduce dimensionality and make data linearly seperable'''
kpca = KernelPCA(n_components=2, kernel='linear', fit_inverse_transform=True, gamma=50)
reduced_data = kpca.fit_transform(open_series)
open_back = kpca.inverse_transform(reduced_data)
pca = PCA()
open_pca = pca.fit_transform(open_series)

cov_matrix = np.cov(reduced_data[:,0], reduced_data[:,1])
print(cov_matrix)
'''KMeans with PCA (reducing data set to 2 columns only)'''
kmeans_pca = KMeans(init='k-means++', n_clusters=6, n_init=10).fit(reduced_data)
print(kmeans_pca.labels_)

plt.scatter(reduced_data[:,0],reduced_data[:,1], c=kmeans_pca.labels_, cmap='rainbow')
plt.show()

'''KMeans with PCA (reducing data set to 3 columns'''
kpca = KernelPCA(n_components=3, kernel='linear', fit_inverse_transform=True, gamma=50)
reduced_data_2 = kpca.fit_transform(open_series)
kmeans_pca_2 = KMeans(init='k-means++', n_clusters=6, n_init=10).fit(reduced_data_2)

fig = plt.figure()
plt.clf()
ax = Axes3D(fig)
ax.scatter(reduced_data_2[:,0], reduced_data_2[:,1], reduced_data_2[:,2], cmap=plt.cm.nipy_spectral, edgecolor='k')
plt.show()
'''Write post-formatted data into a csv file'''
# with open ("data_processed/open.csv", "wb") as w_file:
#     w = csv.DictWriter(w_file, comp_symbols)
#     for key, value in actual.items():
#         row = {key: actual[key]['open']}
#         w.writerow(row)

print("--- %s seconds ---" % (time.time() - start_time))