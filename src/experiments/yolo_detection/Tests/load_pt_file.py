import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from roboflow import Roboflow
import open3d as o3d
import pygetwindow as gw
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

def yolo_add_boxes(image_file, output_image_path):
    # Image path
    #image_file = "C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/output_bev_image_083.png"
    #image = cv2.imread(image_file)

    # Load the model
    # model = torch.load("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/person_lidar_yolov8m.pt",
    #     weights_only=False)    
    # model.eval()
    model = YOLO("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/person_lidar_yolov8m.pt")

    # Make a prediction
    results = model.predict(image_file, save=False, imgsz=640)
    
    # Extract boxes
    boxes = results[0].boxes  # Obtenemos el objeto de las cajas
    
    # Extraer coordenadas, clases y confidencias
    if boxes is not None:
        for box in boxes:
            # Convertir a listas para manipular más fácilmente
            coords = box.xyxy[0].cpu().numpy().tolist()
            cls = int(box.cls[0])  # Clase
            conf = float(box.conf[0])  # Confianza
            print(f"Clase: {cls}, Confianza: {conf:.2f}, Coordenadas: {coords}")
    
    # Print the results for debugging
    print(results)

if __name__ == "__main__":
    yolo_add_boxes("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/local_test/output_bev_image_083.png", "C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/local_test/output_bev_image_083_with_boxes.png")