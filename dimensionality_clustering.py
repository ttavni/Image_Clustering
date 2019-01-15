import umap
import hdbscan

def UMAP_reduction(data,min_dist=0.0, n_components=2, n_neighbors=3):

	umap_array = umap.UMAP(
		n_neighbors=n_neighbors,
		min_dist=min_dist,
		n_components=n_components,
		random_state=42,
	).fit_transform(data)

	return umap_array

def HDBSCAN_clustering(data,min_samples=5,min_cluster_size=5):

	labels = hdbscan.HDBSCAN(
		min_samples=min_samples,
		min_cluster_size=min_cluster_size,
	).fit_predict(data)

	return labels



