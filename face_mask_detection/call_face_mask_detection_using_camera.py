from flask import request
import cv2
import json
import requests
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import rectangle

cap = cv2.VideoCapture(0)
content_type = 'image/jpeg'
headers = {'content-type': content_type}

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', frame)
    # send http request with image and receive response
    response = requests.post('http://127.0.0.1:5000/api/test', data=img_encoded.tostring(), headers=headers)
    # decode response
    #print(response)
    #print(json.loads(response.text))
    jsdata = json.loads(response.text)
    if(len(jsdata)>0):
        for j in jsdata:
            label = j['label']
            x = int(j['x'])
            y = int(j['y'])
            x2 = int(j['x2'])
            y2 = int(j['y2'])
            mask = float(j['mask'])
            withoutMask = float(j['withoutMask'])
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            #cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            rectangle(frame, (x, y), (x2, y2), (0,0,255), 1)

    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()