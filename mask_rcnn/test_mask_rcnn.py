# example of inference with a pre-trained coco model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import numpy as np
#from mrcnn.visualize import display_instances

#from matplotlib import pyplot
#from matplotlib.patches import Rectangle
import cv2
# draw an image with detected objects
def draw_image_with_boxes(pic, pred):
#     # load the image
#     data = pyplot.imread(filename)
#     # plot the image
#     pyplot.imshow(data)
#     # get the context for drawing boxes
#     ax = pyplot.gca()
#     # plot each box
     class_id = pred['class_ids']
     boxes_list = pred['rois']
     obj = class_names[class_id[0]]
     mask = pred['masks']
     mask_reshaped = np.reshape(mask,(mask.shape[0],mask.shape[1]))
     color = (0, 255, 0)
     for box in boxes_list:
#          # get coordinates
          y1, x1, y2, x2 = box
#          # calculate width and height of the box
          #width, height = x2 - x1, y2 - y1
          cv2.rectangle(pic, (x1, y1), (x2, y2), (0,0,255), 1)
          cv2.putText(pic, obj, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
#          # create the shape
          #rect = Rectangle((x1, y1), width, height, fill=False, color='red')
#          # draw the box
          #ax.add_patch(rect)
#     # show the plot
     pic = add_mask(pic,mask_reshaped)

     return (pic)
#     pyplot.show()

def add_mask(img,mask_reshaped,color = [255,0,0],alpha = 0.5):
    for c in range(3):
        img[:, :, c] = np.where(mask_reshaped == True,
                              img[:, :, c] *
                            (1 - alpha) + alpha * color[c] * 255,
                            img[:, :, c])
    return img

# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
# load photograph
#imgPic = load_img('bird.jpg')
#img = img_to_array(imgPic)
pic = cv2.imread('bird.jpg')
dst = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

# make prediction
results = rcnn.detect([dst], verbose=0)
pic_mask_boxes = draw_image_with_boxes(pic, results[0])


np.save('mask_rcnn_mask.npy',results[0]['masks'],)
cv2.imshow('image',pic)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print(results)
# visualize the results
#draw_image_with_boxes('bird.jpg', results[0]['rois'])