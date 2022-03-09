def move_image(image,id,labelID):

	path = CLUSTERING_RESULT_PATH+'/persion'+str(labelID)
	if os.path.exists(path) == False:
		os.mkdir(path)

	filename = str(id) +'.jpg'

	cv2.imwrite(os.path.join(path , filename), image)

	return
