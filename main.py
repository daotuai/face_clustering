# import libraries
import streamlit as st
from PIL import Image
import os
import cv2
import face_recognition
import pickle
from move_image import move_image
import numpy as np
from sklearn.cluster import DBSCAN
from imutils import build_montages

FACE_DATA_PATH = os.path.join(os.getcwd(),'face_cluster')
ENCODINGS_PATH = os.path.join(os.getcwd(),'encodings.pickle')
CLUSTERING_RESULT_PATH = os.getcwd()

imagePaths = st.sidebar.file_uploader(
    "Up load file", accept_multiple_files=True, type=['png', 'jpg', 'jpeg']
)
data = []


for (i, imagePath) in enumerate(imagePaths):
	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	st.sidebar.write("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
	st.sidebar.write(imagePath)

	# loading image to BGR
	image = cv2.imread(imagePath)
	# ocnverting image to RGB format
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(image, model="cnn")
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(image, boxes)
	# build a dictionary of the image path, bounding box location,
	# and facial encodings for the current image
	d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
		for (box, enc) in zip(boxes, encodings)]
	data.extend(d)

# dump the facial encodings data to disk
st.write("[INFO] serializing encodings...")
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
st.write("Encodings of images saved in {}".format(ENCODINGS_PATH))

st.write("[INFO] loading encodings...")
# data = pickle.loads(open("encodings.pickle", "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

# cluster the embeddings
st.write("[INFO] clustering...")

# creating DBSCAN object for clustering the encodings with the metric "euclidean"
clt = DBSCAN(eps=0.5, min_samples=1, metric="euclidean", n_jobs=-1)
clt.fit(encodings)

# determine the total number of unique faces found in the dataset
# clt.labels_ contains the label ID for all faces in our dataset (i.e., which cluster each face belongs to).
# To find the unique faces/unique label IDs, used NumPy’s unique function.
# The result is a list of unique labelIDs
labelIDs = np.unique(clt.labels_)

# we count the numUniqueFaces . There could potentially be a value of -1 in labelIDs — this value corresponds
# to the “outlier” class where a 128-d embedding was too far away from any other clusters to be added to it.
# “outliers” could either be worth examining or simply discarding based on the application of face clustering.


numUniqueFaces = len(np.where(labelIDs > -1)[0])
st.write("[INFO] # unique faces: {}".format(numUniqueFaces))

# loop over the unique face integers
for labelID in labelIDs:
	# find all indexes into the `data` array that belong to the
	# current label ID, then randomly sample a maximum of 25 indexes
	# from the set
	st.write("[INFO] faces for face ID: {}".format(labelID))
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
		move_image(image, i, labelID)

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
	cv2.imwrite(os.path.join(CLUSTERING_RESULT_PATH, title + '.jpg'), montage)



