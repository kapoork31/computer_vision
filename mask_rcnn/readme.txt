mask-rcnn

- belonging from the rcnn family
- state of the art framework in 2019
- mask-rcnn involves object detection and mask-rcn.
- mask-rcnn will output the object that exists, the bounding box of the object, and the mask of the object so we can segment the object

1. We take an image as input and pass it to the ConvNet, which returns the feature map for that image
2. Region proposal network (RPN) is applied on these feature maps. This returns the object proposals along with their objectness score
3. A RoI pooling layer is applied on these proposals to bring down all the proposals to the same size
4. Finally, the proposals are passed to a fully connected layer to classify and output the bounding boxes for objects. It also returns the mask for each proposal

understanding mask R-CNN
- backbone model- use ResNet 101 architecture to extract features from the images in mask r-cnn.
- region proposal network(RPN) - take these feature maps obtained in the previous step and a rpn. 
 - is used to predict if whether an object is present in that proposed region.
- region of interest - a pooling layer is used to make all the RPN's the same size. 
 - these ROI's are then passed through a fully connected network to get class predictions and bounding box predictions.
- as usual, use no maxima supporession to remove ROI's which overlap with each other.
 - so we have the ground truth box for the object, using IOU(area of intersection/area of the union), if IOU is > 0.5
 - then retain that ROI, otherwise do not retain them.
- segmentation mask - returns mask of whole image where true exists if that comparitive pixel is part of the object.

- to implement mask r-cnn
 - clone the mask-rcnn github repo
 - since version of tensorflow, keras may vary with your base values
 - setup a pipenv and set up a pipenv where you have cloned this repo
 - then run pipenv install, will convert requirements.txt to a pipenv file and download the libraries
 - also run python setup.py to create the mask-rcnn library
 - need to download the model weights file as well., -https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5



