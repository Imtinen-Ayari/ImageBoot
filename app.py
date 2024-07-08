from flask import Flask, request, jsonify
import cv2
import numpy as np
import io
import base64
from PIL import Image
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_yolo_model():
    weights_path = os.path.join(BASE_DIR, "yolov3.weights")
    config_path = os.path.join(BASE_DIR, "yolov3.cfg")
    names_path = os.path.join(BASE_DIR, "coco.names")

    if not os.path.exists(weights_path) or not os.path.exists(config_path) or not os.path.exists(names_path):
        raise FileNotFoundError("Les fichiers YOLO nécessaires ne sont pas trouvés.")

    net = cv2.dnn.readNet(weights_path, config_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def detect_objects(image, net, output_layers):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, class_ids, confidences

def extract_car(image, boxes, class_ids, classes):
    height, width = image.shape[:2]
    for i in range(len(boxes)):
        if classes[class_ids[i]] == 'car':
            x, y, w, h = boxes[i]
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            car = image[y:y+h, x:x+w]
            return car
    return None

def align_cars(car1, car2):
    if car1 is None or car2 is None or car1.size == 0 or car2.size == 0:
        raise ValueError("L'une des voitures fournies est vide.")
    
    gray1 = cv2.cvtColor(car1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(car2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) > 10:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)
        aligned_car2 = cv2.warpAffine(car2, H, (car1.shape[1], car1.shape[0]))
        return aligned_car2
    else:
        raise ValueError("Pas assez de correspondances trouvées pour aligner les images.")

def compare_images(image1, image2, output_path1, output_path2):
    net, classes, output_layers = load_yolo_model()

    boxes1, class_ids1, confidences1 = detect_objects(image1, net, output_layers)
    boxes2, class_ids2, confidences2 = detect_objects(image2, net, output_layers)

    car1 = extract_car(image1, boxes1, class_ids1, classes)
    car2 = extract_car(image2, boxes2, class_ids2, classes)

    if car1 is None or car1.size == 0:
        raise ValueError("Aucune voiture détectée ou image vide dans l'image 1.")
    if car2 is None or car2.size == 0:
        raise ValueError("Aucune voiture détectée ou image vide dans l'image 2.")
    
    car2_aligned = align_cars(car1, car2)

    gray_car1 = cv2.cvtColor(car1, cv2.COLOR_BGR2GRAY)
    gray_car2 = cv2.cvtColor(car2_aligned, cv2.COLOR_BGR2GRAY)
    gray_car1 = cv2.GaussianBlur(gray_car1, (5, 5), 0)
    gray_car2 = cv2.GaussianBlur(gray_car2, (5, 5), 0)
    
    difference = cv2.absdiff(gray_car1, gray_car2)
    _, thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    car1_with_contours = car1.copy()
    car2_with_contours = car2_aligned.copy()
    cv2.drawContours(car1_with_contours, contours, -1, (0, 0, 255), 2)
    cv2.drawContours(car2_with_contours, contours, -1, (0, 0, 255), 2)

    cv2.imwrite(output_path1, car1_with_contours)
    cv2.imwrite(output_path2, car2_with_contours)

@app.route('/compare', methods=['POST'])
def compare_images_route():
    try:
        file1 = request.files['image1']
        file2 = request.files['image2']
        
        image1 = Image.open(io.BytesIO(file1.read()))
        image2 = Image.open(io.BytesIO(file2.read()))
        
        image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

        output1, output2 = 'output1.jpg', 'output2.jpg'
        compare_images(image1, image2, output1, output2)

        with open(output1, 'rb') as f:
            img1_base64 = base64.b64encode(f.read()).decode('utf-8')
        with open(output2, 'rb') as f:
            img2_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return jsonify({'image1': img1_base64, 'image2': img2_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
