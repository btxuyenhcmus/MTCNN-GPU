import requests
import json
import base64
import cv2

cap = cv2.VideoCapture(0)

retval, image = cap.read()
count = 0

while True:
    retval, image = cap.read()
    # Display the resulting frame
    try:
        cv2.imshow('preview',image)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    except:
        continue
    # if count == 20:
    count = 0
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    headers = {
        'Content-Type': r"application/json"
    }
    data = {
        'jpg_as_text': jpg_as_text
    }
    response = requests.post(url='http://localhost:3000/api/detect-face', headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        continue
    data = json.loads(response.text)
    print(data)
    # count += 1