from src import show_bboxes
from PIL import Image
import numpy as np
import requests
import json
import base64
import cv2

cap = cv2.VideoCapture(0)
cv2.namedWindow('preview', cv2.WINDOW_GUI_EXPANDED)

while True:
    retval, image = cap.read()
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    try:
        headers = {
            'Content-Type': r"application/json"
        }
        data = {
            'jpg_as_text': jpg_as_text
        }
        response = requests.post(
            url='http://localhost:3000/api/detect-face', headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            continue
        data = json.loads(response.text)
        bounding_boxes = data.get('bounding_boxes', [[]])
        landmarks = data.get('landmarks', [[]])
        PIL_image = Image.fromarray(image.astype('uint8'), 'RGB')
        image = show_bboxes(PIL_image, bounding_boxes, landmarks)
        image = np.array(image)
        cv2.imshow('preview', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    except:
        cv2.imshow('preview', image)
