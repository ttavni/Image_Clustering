import os
import numpy as np
import pandas as pd
from image_feature_extraction import ExtractImageFeature
from dimensionality_clustering import UMAP_reduction
from dimensionality_clustering import HDBSCAN_clustering
import bs4
from copy_files import recursive_copy_files

if __name__ == "__main__":

	# DEPENDENCY
	directory_of_images = 'fruits'
	image_type = '.jpg'

	# Find all images
	relative_image_directory = 'images/{}/'.format(directory_of_images)

	all_imgs = [(relative_image_directory + x) for x in os.listdir(relative_image_directory) if image_type in x]
	raw_paths = [x.split('/')[-1] for x in all_imgs]

	# Get features from VGG16
	feature_lists = [ExtractImageFeature(img) for img in all_imgs]
	image_embeddings = np.vstack(feature_lists)

	# Reduce dimensions and find cluster labels
	umap_array = UMAP_reduction(image_embeddings)
	cluster_labels = HDBSCAN_clustering(umap_array)

	# Get dataframe of results
	df_data = pd.DataFrame({'URL': raw_paths,'X': umap_array[:,0],'Y': umap_array[:,1],'labels':cluster_labels})

	length_before_outliers = len(df_data)
	df_data = df_data[np.abs(df_data.X - df_data.X.mean()) <= (2 * df_data.X.std())]
	df_data = df_data[np.abs(df_data.Y - df_data.Y.mean()) <= (2 * df_data.Y.std())]
	print('Number of outliers removed: {}'.format(length_before_outliers-len(df_data)))


	# Save data and visualisation
	visualisation_path = 'visualisations/{}/'.format(directory_of_images)

	if not os.path.exists(visualisation_path):
		os.makedirs(visualisation_path)

	df_data.to_csv('{}data.csv'.format(visualisation_path))

	# Load and create index.html file
	with open("visualisations/main/imaged3.html") as inf:
		txt = inf.read()
		soup = bs4.BeautifulSoup(txt, "html.parser")

	with open("{}index.html".format(visualisation_path), "w") as outf:
		outf.write(str(soup))

	recursive_copy_files(relative_image_directory, '{}images/'.format(visualisation_path))






