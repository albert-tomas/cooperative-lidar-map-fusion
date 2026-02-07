import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from roboflow import Roboflow
import open3d as o3d
import pygetwindow as gw

# This file contains functions to create a BEV image from a CSV file, add YOLO boxes to the image, and remove points from the CSV based on the YOLO predictions

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')    

def create_bev_image(csv_file, output_image_path, res=0.02, x_range=(-5, 5), y_range=(-5, 5), z_range=(-2, 2)):
    # Load data from CSV
    df = pd.read_csv(csv_file)

    # Define the grid indices
    x_bins = np.arange(x_range[0], x_range[1], res)
    y_bins = np.arange(y_range[0], y_range[1], res)

    # Histogram computation
    x_indices = np.digitize(df['x(m)'], x_bins) - 1
    y_indices = np.digitize(df['y(m)'], y_bins) - 1

    # Filter points within specified range
    valid_points = (x_indices >= 0) & (x_indices < len(x_bins)) & \
                (y_indices >= 0) & (y_indices < len(y_bins)) & \
                (df['z(m)'] >= z_range[0]) & (df['z(m)'] <= z_range[1])
    # TODO: we could invert the order, first filter and then digitize, to make it faster

    # Create an empty image
    bev_image = np.zeros((len(y_bins), len(x_bins)), dtype=np.uint8)

    # Accumulate points in the image array
    np.add.at(bev_image, (y_indices[valid_points], x_indices[valid_points]), 1)

    # Plot the image
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(bev_image, cmap='gray', origin='lower')

    # Remove axes and padding
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Save the figure
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def yolo_add_boxes(image_file, output_image_path):
    # Image path
    #image_file = "C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/output_bev_image_083.png"
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
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # BGR, 2 is the line thickness

        # Draw the class label and confidence
        label = f"{prediction['class']} ({prediction['confidence']*100:.2f}%)"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image with the predictions
    cv2.imwrite(output_image_path, image)

    # Print success message
    print("Image saved with predictions.")
    
    return predictions

def create_xy_grid(x_range=(-5, 5), y_range=(-5, 5), step=1.0, z_level=0.0):
    lines = []
    points = []
    # vertical lines
    for x in np.arange(x_range[0], x_range[1]+step, step):
        points.append([x, y_range[0], z_level])
        points.append([x, y_range[1], z_level])
        lines.append([len(points)-2, len(points)-1])
    # horizontal lines
    for y in np.arange(y_range[0], y_range[1]+step, step):
        points.append([x_range[0], y, z_level])
        points.append([x_range[1], y, z_level])
        lines.append([len(points)-2, len(points)-1])
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines  = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector([[0.5,0.5,0.5] for _ in lines])
    return grid

def show_csv(file_path):
    # 1. Cargo CSV y creo PointCloud
    df = pd.read_csv(file_path, usecols=["x(m)","y(m)","z(m)"])
    pts = np.vstack((df["x(m)"], df["y(m)"], df["z(m)"])).T
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    # 2. Creo rejilla y ejes
    grid = create_xy_grid(x_range=(-10,10), y_range=(-10,10), step=1.0)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])

    # 3. Dibujo todo junto
    o3d.visualization.draw_geometries(
        [pc, grid, axes],
        window_name="Point Cloud with Grid and Axes",
        width=800, height=600
    )


def remove_points_in_boxes(csv_path, predictions, output_csv_path, res=0.02, x_range=(-5, 5), y_range=(-5, 5)):
    # Load the original CSV
    df = pd.read_csv(csv_path)

    # Original BEV image dimensions
    bev_width = int((x_range[1] - x_range[0]) / res)
    bev_height = int((y_range[1] - y_range[0]) / res)

    # Assumed YOLO image size (change if you used a different size!)
    yolo_width = 600
    yolo_height = 600

    scale_x = bev_width / yolo_width
    scale_y = bev_height / yolo_height

    for pred in predictions['predictions']:
        # Box coordinates in pixels
        x1_px = int((pred['x'] - pred['width'] / 2) * scale_x)
        x2_px = int((pred['x'] + pred['width'] / 2) * scale_x)
        y1_px = int((pred['y'] - pred['height'] / 2) * scale_y)
        y2_px = int((pred['y'] + pred['height'] / 2) * scale_y)
        
        # Invert Y axis to match BEV orientation (origin at bottom-left)  # <--- cambiado
        y1_idx = bev_height - y2_px  # y2_px is higher in image, so becomes y_min in BEV
        y2_idx = bev_height - y1_px  # y1_px is lower in image, so becomes y_max in BEV


        # Convert pixels to real-world coordinates
        # Note: the Y-axis in the image grows downward, and in BEV it does too, so no need to invert it
        x_min = x_range[0] + x1_px * res
        x_max = x_range[0] + x2_px * res
        y_min = y_range[0] + y1_idx * res
        y_max = y_range[0] + y2_idx * res
        
        print(f"Box coordinates in pixels: {x1_px}, {x2_px}, {y1_px}, {y2_px}")
        print(f"Converted to real-world coordinates: {x_min:.2f}, {x_max:.2f}, {y_min:.2f}, {y_max:.2f}")

        inside = df[((df['x(m)'] >= x_min) & (df['x(m)'] <= x_max) &
            (df['y(m)'] >= y_min) & (df['y(m)'] <= y_max))]
        print(f"Points to remove in box {x_min:.2f}-{x_max:.2f}, {y_min:.2f}-{y_max:.2f}: {len(inside)}")

        # Filter out points inside the box (keep only the ones outside)
        df = df[~((df['x(m)'] >= x_min) & (df['x(m)'] <= x_max) &
                (df['y(m)'] >= y_min) & (df['y(m)'] <= y_max))]

    # Save the modified CSV
    df.to_csv(output_csv_path, index=False)
    print(f"New CSV saved without points inside the boxes: {output_csv_path}")

def remove_points_in_box_manually(csv_input_path, xmin, xmax, ymin, ymax, csv_output_path):
    # Load the original CSV
    df = pd.read_csv(csv_input_path)

    # Filter out points inside the box (keep only the ones outside)
    df = df[~((df['x(m)'] >= xmin) & (df['x(m)'] <= xmax) &
            (df['y(m)'] >= ymin) & (df['y(m)'] <= ymax))]

    # Save the modified CSV if output path is provided
    if csv_output_path:
        df.to_csv(csv_output_path, index=False)
        print(f"New CSV saved without points inside the box: {csv_output_path}")

if __name__ == "__main__":
    #clear_terminal()
    # Convert initial CSV to PNG
    # Show CSV
    # show_csv("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/solo_test/solo_person.csv")
    # # create_bev_image('C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/solo_test/solo_person.csv', 'C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/solo_test/output_bev_solo_person.png')
    # # # # Detect objects from PNG with YOLO
    # predictions = yolo_add_boxes("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/solo_test/output_bev_solo_person.png",
    #                     "C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/solo_test/output_bev_solo_person_with_boxes.png")
    # # # print(predictions['predictions'])
    # # # # Remove points from CSV
    # remove_points_in_boxes('C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/solo_test/solo_person.csv', predictions, 'C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/solo_test/solo_person_removed.csv')
    # show_csv("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/solo_test/solo_person_removed.csv")
    # # remove_points_in_boxes('C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/frame_083.csv', predictions, 'C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/frame_083_no_person.csv')
    # # show_csv("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/frame_083_no_person.csv")
    
    
    # remove_points_in_box_manually("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/solo_test/solo_person.csv", 0, 1, 1, 2, "C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/solo_test/solo_person_filtered.csv")
    # show_csv("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/solo_test/solo_person_filtered.csv")
    
    
    show_csv("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/final_test/frame_083.csv")
    create_bev_image('C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/final_test/frame_083.csv', 'C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/final_test/output_bev_image_083.png')
    # # # Detect objects from PNG with YOLO
    predictions = yolo_add_boxes("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/final_test/output_bev_image_083.png",
                        "C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/final_test/output_bev_image_083_with_boxes.png")
    # # print(predictions['predictions'])
    # # # Remove points from CSV
    remove_points_in_boxes('C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/final_test/frame_083.csv', predictions, 'C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/final_test/frame_083_removed.csv')
    show_csv("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/final_test/frame_083_removed.csv")