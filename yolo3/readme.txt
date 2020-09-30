Application of the yolo object detection algorithm using opencv

Opencv: also opencv has a deep learning framework that works with YOLO. 
Just make sure you have opencv 3.4.2 at least. 
Advantage: it works without needing to install anything except opencv. 
Disadvantage: it only works with CPU, so you can’t get really high speed to process videos in real time.

clone this repo : https://github.com/pjreddie/darknet
- possesses the config file, and coco.names file which lists the names of the objects the model can predict
- wget https://pjreddie.com/media/files/yolov3.weights to get the weight file for this, is over 200mb

# the image is split up into its grid which is what yolo does
# produces an output vector for each grid
# first 5 values are the vector each grid in an image gets from yolo are as follows:
# object of any class present confidence, x,y(center of bounding box) , height, weight of bounding box.
# after thats its the prediction of whether a specific object exists assigned to the class names from the coco.names file
# bird is the 15th name in the list
# so the 19th element in the vector will refer to the probability of a bird existing in the picture.

# using no maxima suppressions we define a list of proposal regions to examine
# then draw a box in this proposed region and add a label of the object predicted.
