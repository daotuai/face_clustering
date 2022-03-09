# USAGE
# python cluster_faces.py --encodings encodings.pickle

# import the necessary packages
# DBSCAN model for clustering similar encodings
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2
import shutil
import os
from constants import FACE_DATA_PATH, ENCODINGS_PATH, CLUSTERING_RESULT_PATH


def move_image(image,id,labelID):

	path = CLUSTERING_RESULT_PATH+'/persion'+str(labelID)
	if os.path.exists(path) == False:
		os.mkdir(path)

	filename = str(id) +'.jpg'
	cv2.imwrite(os.path.join(path , filename), image)
	return


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of parallel jobs to run (-1 will use all CPUs)")
args = vars(ap.parse_args())

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

# cluster the embeddings
print("[INFO] clustering...")

# creating DBSCAN object for clustering the encodings with the metric "euclidean"
clt = DBSCAN(eps=0.5, min_samples=1, metric="euclidean", n_jobs=args["jobs"])
clt.fit(encodings)

labelIDs = np.unique(clt.labels_)

numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

# loop over the unique face integers
for labelID in labelIDs:
	print("[INFO] faces for face ID: {}".format(labelID))
	idxs = np.where(clt.labels_ == labelID)[0]
	idxs = np.random.choice(idxs, size=min(25, len(idxs)),
		replace=False)

	# initialize the list of faces to include in the montage
	faces = []

	# loop over the sampled indexes
	for i in idxs:
		# load the input image and extract the face ROI
		image = cv2.imread(data[i]["imagePath"])
		(top, right, bottom, left) = data[i]["loc"]
		face = image[top:bottom, left:right]

		# puting the image in the clustered folder
		move_image(image,i,labelID)

		# force resize the face ROI to 96mx96 and then add it to the
		# faces montage list
		face = cv2.resize(face, (96, 96))
		faces.append(face)

	# create a montage using 96x96 "tiles" with 5 rows and 5 columns
	montage = build_montages(faces, (96, 96), (5, 5))[0]
	
	# show the output montage
	title = "Face ID #{}".format(labelID)
	title = "Unknown Faces" if labelID == -1 else title
	"""
	cv2.imshow(title, montage)
	cv2.waitKey(0)
	"""
	cv2.imwrite(os.path.join(CLUSTERING_RESULT_PATH, title+'.jpg'), montage)
