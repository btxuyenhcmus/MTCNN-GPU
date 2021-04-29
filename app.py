from PIL import Image
from flask import Flask, request, jsonify
from src import detect_faces
from io import BytesIO
import base64
import json


app = Flask(__name__)


def convert_img2pilimg(img):
    if type(img) is str:
        imgData = base64.b64decode(img)
        image = Image.open(BytesIO(imgData))
        return image
    return img


@app.route('/api/detect-face', methods=['POST'])
def detect_face():
    json_data = request.get_json()
    jpg_as_text = json_data.get('jpg_as_text', '')
    img = convert_img2pilimg(jpg_as_text)
    bounding_boxes, landmarks = detect_faces(img)
    data = {
        'bounding_boxes': bounding_boxes.tolist() if not type(bounding_boxes) is list else [],
        'landmarks': landmarks.tolist() if not type(landmarks) is list else []
    }
    return jsonify(data)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000)
