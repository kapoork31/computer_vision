# face_mask_detection

- firstly find the face using opencv python with the haar caascade face classifier
- returns a bounding box where the face exists
- then inside this box, classify whether its got a face without mask or with mask
	- classifier is the mobile vnet 2 which is useful on less powerful machines like a basic cpu laptop or a mobile etc.
	- model has been trained and the h5 model is saved in this repo
	- model trained using transfer learning. uses the mobile v2 as the base model.
	- training data is around 650 images of each label, useing an imageDataGenerator in keras, data set is augmented with zooms, rotations etc to help generalization.
	- attached an average pooling layer, a fully connected layer and a dense layer with 2 outputs, whether face mask or not
	- since mobile v2 has learned general features that can be used in distinguishing between our training data
	- new convolutions are not needed as features are not needed to be relearned.
	- model trained in google colab as free gpu use, 20 epochs, takes a few minutes
	- if trained on my potato of a pc, would take hours.
	
	
- training_model_.ipynb is a file where transfer learning happens
- face_recog_haar_mobile_v2_as_flask_service.ipynb is where the detection service is set up as a service on local network
- call_face_mask_detection_using_camera.ipynb is file where we call this service


