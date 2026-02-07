from datetime import datetime
import os
import pandas as pd
import open3d as o3d
import numpy as np
import time
import pygetwindow as gw
from roboflow import Roboflow
import matplotlib.pyplot as plt
import cv2


# This file is like alignment_v18 but removed all the alignment stuff, just to test the BEV image generation and YOLO detection
# v6 - Added BEV_CONFIG


# File to be processed
FILE_NAME = {
    "catture": "D:/LiDAR-captures/Capture1911/CSV",
    "strada1": "D:/LiDAR-captures/strada1/CSV",
    "strada2": "D:/LiDAR-captures/strada2/CSV",
    "strada3": "D:/LiDAR-captures/strada3/CSV"
}

# Configuration
CONFIG = {
    "selected_file": "strada2", # Change this to the desired file
    "max_number_of_clouds": 200,
    "voxel_size": 0.1,
    "print_realtime": False,
    #"folder_path_csv": "C:/Users/Albert/Desktop/lidar-fused-3d-map/CSVfiles",
    "save_map": False,
    # "save_trajectory": False,
    "trajectory_sphere": True, # If True, the trajectory will be visualized as a sphere (for better visualization)
    "radius": 0.2, # Radius for filtering close points
}

BEV_CONFIG = {
    "res": 0.02,
    "x_range": (-5, 5),
    "y_range": (-5, 5),
    "z_range": (-2, 2),
    "scale": 2,  # Scale factor for the BEV image
}

# Best values: 
# BEV_CONFIG = {
#     "res": 0.02,
#     "x_range": (-5, 5),
#     "y_range": (-5, 5),
#     "z_range": (-2, 2),
#     "scale": 6,  # Scale factor for the BEV image
# }

COLORS = {
    "RED": [1, 0, 0],
    "GREEN": [0, 1, 0],
    "BLUE": [0, 0, 1],
    "GREY": [0.5, 0.5, 0.5],
}

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_csv_files(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the folder.")
    else:
        print(f"Found {len(csv_files)} CSV files.")
    return csv_files

def setup_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    window = gw.getWindowsWithTitle('Open3D')[0]
    window.maximize()
    return vis


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

    # Create an empty image
    bev_image = np.zeros((len(y_bins), len(x_bins)), dtype=np.uint8)

    # Accumulate points in the image array
    np.add.at(bev_image, (y_indices[valid_points], x_indices[valid_points]), 1)

    # Plot the image
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    bev_image = np.clip(bev_image * BEV_CONFIG["scale"], 0, 255)
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

def initialize_roboflow():
    # Initialize Roboflow
    rf = Roboflow(api_key="m3mVIcAJXOFxaob0eVAt")

    # Load the trained model
    model = rf.workspace("sicariata").project("person_lidar").version(1).model
    return model

def yolo_add_boxes(image_file, output_image_path, model):
    
    # Load the input image using OpenCV
    image = cv2.imread(image_file)

    # Run the model prediction (Roboflow format)
    predictions = model.predict(image, confidence=30, overlap=30).json()

    # Draw bounding boxes and labels on the image
    for prediction in predictions['predictions']:
        # Calculate top-left and bottom-right coordinates
        x1 = int(prediction['x'] - prediction['width'] / 2)
        y1 = int(prediction['y'] - prediction['height'] / 2)
        x2 = int(prediction['x'] + prediction['width'] / 2)
        y2 = int(prediction['y'] + prediction['height'] / 2)

        # Draw the bounding box in green
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Compose and draw the label with confidence
        label = f"{prediction['class']} ({prediction['confidence']*100:.2f}%)"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image with the drawn detections
    cv2.imwrite(output_image_path, image)

    return predictions


def icp_alignment(csv_files, folder_path, vis_map):
    max_number_of_clouds = CONFIG["max_number_of_clouds"]

    sel = CONFIG["selected_file"]
    base_dir = f"D:/LiDAR-captures/{sel}/bev_images_test"
    
    # Initialize the Roboflow model
    model = initialize_roboflow()
    
    # Load first cloud and create BEV + predictions
    first_csv = csv_files[0]
    first_bev = os.path.join(base_dir, "image_0.png")
    create_bev_image(
        os.path.join(folder_path, first_csv), first_bev,
        res=BEV_CONFIG["res"], x_range=BEV_CONFIG["x_range"], y_range=BEV_CONFIG["y_range"], z_range=BEV_CONFIG["z_range"]
    )
    predictions = yolo_add_boxes(first_bev, os.path.join(base_dir, "image_0_boxes.png"), model)
    

    # Iterate over remaining clouds
    for i in range(1, min(len(csv_files), max_number_of_clouds)):
        # Load cloud and create BEV + predictions
        csv = csv_files[i]
        bev = os.path.join(base_dir, f"image_{i}.png")
        create_bev_image(os.path.join(folder_path, csv), bev,
                        res=BEV_CONFIG["res"], x_range=BEV_CONFIG["x_range"], y_range=BEV_CONFIG["y_range"], z_range=BEV_CONFIG["z_range"])
        predictions = yolo_add_boxes(bev, os.path.join(base_dir, f"image_{i}_boxes.png"), model)
        
        
        print(f"Image {i} saved.")
    
    return None, None, None    

def save_static_map(global_map, timestamp, base_path):
    output_path = os.path.join(base_path, f"{timestamp}_map.ply")
    o3d.io.write_point_cloud(output_path, global_map)
    print(f"Static map saved to {output_path}")

def save_trajectory(global_trajectory, timestamp, base_path):
    output_path = os.path.join(base_path, f"{timestamp}_trajectory.ply")
    o3d.io.write_point_cloud(output_path, global_trajectory)
    print(f"Trajectory map saved to {output_path}")

def save_person_cloud(person_global_cloud, timestamp, base_path):
    output_path = os.path.join(base_path, f"{timestamp}_person.ply")
    o3d.io.write_point_cloud(output_path, person_global_cloud)
    print(f"Person map saved to {output_path}")

def main():
    start_time = time.time()

    # Get CSV file list from selected folder
    folder_path = FILE_NAME[CONFIG["selected_file"]]
    csv_files = get_csv_files(folder_path)
    
    # # Set up visualizer if real-time visualization is enabled
    if CONFIG["print_realtime"]:
        vis_map = setup_visualizer()
    else:
        vis_map = None
    
    # Run alignment
    static_map, global_trajectory, global_person_cloud = icp_alignment(csv_files, folder_path, vis_map)
    
    # Prepare save path and timestamp
    timestamp = datetime.now().strftime("%d%m_%H%M")  # Format: daymonth_hourminute
    base_path = f"C:/Users/Albert/Desktop/lidar-fused-3d-map/src/visualization/saved_maps_and_trajectories/{CONFIG['selected_file']}"
    
    # Save results if enabled
    if CONFIG["save_map"]:
        save_static_map(static_map, timestamp, base_path)
        save_trajectory(global_trajectory, timestamp, base_path)
        save_person_cloud(global_person_cloud, timestamp, base_path)
    
    # Print execution time
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    # Show final visualization if enabled
    if CONFIG["print_realtime"]:
        vis_map.run()   


if __name__ == "__main__":
    clear_terminal()
    main()