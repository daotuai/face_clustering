def move_image(image,id,labelID):

	path = CLUSTERING_RESULT_PATH+'/persion'+str(labelID)
	# os.path.exists() method in Python is used to check whether the specified path exists or not.
	# os.mkdir() method in Python is used to create a directory named path with the specified numeric mode.
	if os.path.exists(path) == False:
		os.mkdir(path)

	filename = str(id) +'.jpg'
	# Using cv2.imwrite() method
	# Saving the image

	cv2.imwrite(os.path.join(path , filename), image)

	return