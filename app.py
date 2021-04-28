from PIL import Image
from src import detect_faces
from aiohttp import web
from io import BytesIO
import base64
import socketio


sio = socketio.AsyncServer(cor_allowed_origins='*')
print("____________AsyncServer Runing____________: {}".format(sio))
app = web.Application()
print("____________Application Runing____________: {}".format(app))
sio.attach(app)


def convert_img2pilimg(img):
    if type(img) is str:
        imgData = base64.b64decode(img)
        image = Image.open(BytesIO(imgData))
        return image
    return img


@sio.event
async def connect(sid, environ):
    print('connect ', sid)
    await sio.emit('connected', {'message': 'connect success'}, room=sid)


@sio.on('detection-channel')
async def detect_face(sid, img):
    try:
        img = convert_img2pilimg(img)
        bounding_boxes, landmarks = detect_faces(img)
        data = {
            'bounding_boxes': bounding_boxes.tolist(),
            'landmarks': landmarks.tolist()
        }
        await sio.emit('detection-channel', data=data, room=sid)

    except:
        print("Not yet detected face!!!!!!")

if __name__ == '__main__':
    web.run_app(app)
