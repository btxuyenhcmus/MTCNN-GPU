import base64
import socketio
import cv2

sio = socketio.Client()

# Connect to server
sio.connect('http://localhost:8080')
cap = cv2.VideoCapture(0)


@sio.on('detection-channel')
def message(data):
    try:
        bounding_boxes, landmarks = data.get('bounding_boxes'), data.get('landmarks')
        print(bounding_boxes)
        print(landmarks)
    except:
        print("Cannot receive information from server!!")

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
    if count == 20:
        count = 0
        retval, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        sio.emit('detection-channel', jpg_as_text)
    count += 1