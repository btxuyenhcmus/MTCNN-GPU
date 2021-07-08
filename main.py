from src import detect_faces, show_bboxes, detect_faces_type1, detect_faces_type2,detect_faces_type3
from PIL import Image
img = Image.open('images/family.jpeg')
bounding_boxes, landmarks = detect_faces_type3(img)
show_bboxes(img, bounding_boxes, landmarks)