import sys
import os
import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt

def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
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
            
            # Assurez-vous que les coordonnées de recadrage sont dans les limites de l'image
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            car = image[y:y+h, x:x+w]
            print(f"Extraction des coordonnées : x={x}, y={y}, w={w}, h={h}")
            print(f"Dimensions de l'image extraite : {car.shape}")
            return car
    return None

def align_cars(car1, car2):
    if car1 is None or car2 is None:
        raise ValueError("L'une des voitures fournies est vide.")
    
    if car1.size == 0 or car2.size == 0:
        raise ValueError("L'une des images de voiture fournies est vide après l'extraction.")

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

def compare_images(image_path1, image_path2, output_path1, output_path2):
    net, classes, output_layers = load_yolo_model()
    
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    if image1 is None:
        raise ValueError(f"Erreur lors du chargement de l'image à partir de {image_path1}")
    if image2 is None:
        raise ValueError(f"Erreur lors du chargement de l'image à partir de {image_path2}")

    boxes1, class_ids1, confidences1 = detect_objects(image1, net, output_layers)
    boxes2, class_ids2, confidences2 = detect_objects(image2, net, output_layers)

    print(f"Image 1: {len(boxes1)} objets détectés")
    print(f"Image 2: {len(boxes2)} objets détectés")

    car1 = extract_car(image1, boxes1, class_ids1, classes)
    car2 = extract_car(image2, boxes2, class_ids2, classes)

    print(f"Voiture extraite de l'image 1 : {'Oui' if car1 is not None else 'Non'}")
    print(f"Voiture extraite de l'image 2 : {'Oui' if car2 is not None else 'Non'}")

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
    
    num_differences = len(contours)
    
    if num_differences == 0:
        print("Aucune différence détectée.")
    else:
        print(f"{num_differences} différences détectées.")
    
    car1_with_contours = car1.copy()
    car2_with_contours = car2_aligned.copy()
    cv2.drawContours(car1_with_contours, contours, -1, (0, 0, 255), 2)
    cv2.drawContours(car2_with_contours, contours, -1, (0, 0, 255), 2)

    cv2.imwrite(output_path1, car1_with_contours)
    cv2.imwrite(output_path2, car2_with_contours)
    
url = "http://52.47.71.44:1111/Comparator"
response = requests.post(url, data={'User-Agent': 'Mozilla/5.0'})

# Récupération du contenu de la page
html_page = response.text

if __name__ == "__main__":
    image_path1 = sys.argv[1]
    image_path2 = sys.argv[2]
    output_path1 = sys.argv[3]
    output_path2 = sys.argv[4]
    compare_images(image_path1, image_path2, output_path1, output_path2)
