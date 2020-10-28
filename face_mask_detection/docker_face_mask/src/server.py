import cv2
import numpy as np
import tensorflow.keras.models as models
from flask import Flask, request, Response
from cv2 import CascadeClassifier
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

app = Flask(__name__)
 
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
m = models.load_model("face_mask_model.h5")  
    
@app.route('/api/test', methods=['POST'])
def draw_image_with_boxes():
    # load the image
    r = request
    #print(len(r.data))
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # perform face detection
    bboxes = classifier.detectMultiScale(frame)
    reslist = []
    for box in bboxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        face = frame[y:y2, x:x2]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        (mask, withoutMask) = m.predict(face)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        # display the label and bounding box rectangle on the output
        # frame
        #cv2.putText(frame, label, (x, y - 10),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        res = {'label':str(label),'x':str(x),'y':str(y),'x2':str(x2),'y2':str(y2),'mask':str(mask),'withoutMask':str(withoutMask)}
        reslist.append(res)
    
    jsonres = json.dumps(reslist)
    #print(jsonres)
    return Response(response=jsonres, status=200, mimetype="application/json")
 
if __name__ == "__main__":
    app.run(host='0.0.0.0')