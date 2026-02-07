import cv2
from roboflow import Roboflow

# This file is used to add boxes to the image using the Roboflow API.
# It loads the image, makes predictions using a trained model, and draws the predicted boxes on the image.

# Image path
image_file = "D:/lidar-thesis/CSV_examples/my_test/output_bev_image_083.png"
image = cv2.imread(image_file)

# Initialize Roboflow
rf = Roboflow(api_key="m3mVIcAJXOFxaob0eVAt")

# Load the trained model
model = rf.workspace("sicariata").project("person_lidar").version(1).model

# Make the prediction
predictions = model.predict(image, confidence=40, overlap=30).json()

# Draw the boxes on the image
for prediction in predictions['predictions']:
    x1 = int(prediction['x'] - prediction['width'] / 2)
    y1 = int(prediction['y'] - prediction['height'] / 2)
    x2 = int(prediction['x'] + prediction['width'] / 2)
    y2 = int(prediction['y'] + prediction['height'] / 2)

    # Draw the rectangle on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # BGR, 2 es el grosor de la línea

    # Draw the class label and confidence
    label = f"{prediction['class']} ({prediction['confidence']*100:.2f}%)"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the image with the predictions
cv2.imwrite("D:/lidar-thesis/CSV_examples/my_test/output_with_boxes.png", image)
print("Imagen guardada con las predicciones.")