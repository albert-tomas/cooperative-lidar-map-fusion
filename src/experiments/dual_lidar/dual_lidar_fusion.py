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
from collections import defaultdict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from icp_alignment.alignment_v21 import *

# This script will perform the fusion of two LiDAR captures using ICP and YOLO for object detection, this version for now only uses one LiDAR capture to generate a single map

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
    "max_number_of_clouds": 50,
    "voxel_size": 0.1,
    "print_realtime": True,
    "save_map": False,
    "realtime_people_only": False, # If True, we will only visualize the dynamic objects (people) in the current frame
    "trajectory_sphere": True, # If True, the trajectory will be visualized as a sphere (for better visualization)
    "radius": 0.2, # Radius for filtering close points
}

BEV_CONFIG = {
    "res": 0.02,
    "x_range": (-5, 5),
    "y_range": (-5, 5),
    "z_range": (-2, 2),
    "scale": 1,  # Scale factor for the BEV image
}

COLORS = {
    "RED": [1, 0, 0],
    "GREEN": [0, 1, 0],
    "BLUE": [0, 0, 1],
    "GREY": [0.5, 0.5, 0.5],
}

def initialize_maps():
    static_map = o3d.geometry.PointCloud()
    trajectory_global_cloud = o3d.geometry.PointCloud()
    person_global_cloud = o3d.geometry.PointCloud()
    show_cloud = o3d.geometry.PointCloud()
    
    return static_map, trajectory_global_cloud, person_global_cloud, show_cloud
    

def icp_initialize():
    # Configuration
        threshold = 0.5  # Distance threshold for ICP matching. Correspondance distance
        max_iterations = 50 # Maximum number of iterations for ICP
        max_number_of_clouds = CONFIG["max_number_of_clouds"]
        voxel_size = CONFIG["voxel_size"]
        sel = CONFIG["selected_file"]
        base_dir = f"D:/LiDAR-captures/{sel}/bev_images"
    
        # ICP setup
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iterations)
        
        # Initialize the voxel counter and max count
        voxel_counter = defaultdict(int)
        max_count = 1
        
        icp_params = {
            "threshold": threshold,
            "max_iterations": max_iterations,
            "voxel_size": voxel_size,
            "base_dir": base_dir,
            "max_number_of_clouds": max_number_of_clouds,
            "sel": sel,
            "criteria": criteria,
            "voxel_counter": voxel_counter,
            "max_count": max_count}
        
        return icp_params

def icp_alignment(csv_files, folder_path, i, input_cloud, static_map, trajectory_global_cloud, person_global_cloud, icp_params, current_transformation, max_count):

    # Load cloud and create BEV + predictions
    current_csv = csv_files[i]
    bev_image_path = os.path.join(icp_params["base_dir"], f"image_{i}.png")
    create_bev_image(
        os.path.join(folder_path, current_csv), bev_image_path,
        res=BEV_CONFIG["res"], x_range=BEV_CONFIG["x_range"], y_range=BEV_CONFIG["y_range"], z_range=BEV_CONFIG["z_range"]
    )
    predictions = yolo_add_boxes(bev_image_path, os.path.join(icp_params["base_dir"], f"image_{i}_boxes.png"), model)
    
    # Extract detected person points and color them
    person_cloud = extract_points_in_boxes(input_cloud, predictions, res=0.02, x_range=(-5, 5), y_range=(-5, 5))
    change_colors(person_cloud, COLORS["GREEN"])
    person_global_cloud += person_cloud
    
    # Build initial static map (gray) and remove person points
    input_cloud_static = remove_points_by_proximity(input_cloud, person_cloud, radius=0.05)
    input_cloud_static_downsampled = input_cloud_static.voxel_down_sample(icp_params["voxel_size"])
    input_cloud_static_downsampled, _ = input_cloud_static_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    change_colors(input_cloud_static_downsampled, COLORS["GREY"])
    
    if CONFIG["trajectory_sphere"]:
            trajectory_cloud = generate_sphere(radius=0.05, point_count=100)
    else:
            trajectory_cloud = generate_single_point()
        
    
    if i == 0:
        static_map = input_cloud_static_downsampled
        max_count = icp_params["max_count"]
        
        change_colors(trajectory_cloud, COLORS["BLUE"])
        trajectory_global_cloud = trajectory_cloud
        
        
        return predictions, input_cloud_static_downsampled, trajectory_cloud, person_cloud, current_transformation, static_map, max_count
    
    
    
    # Apply the current transformation to the input clouds (for better ICP initialization)
    input_cloud_static_downsampled.transform(current_transformation)
    person_cloud.transform(current_transformation)
    trajectory_cloud.transform(current_transformation)
    
    # Run ICP between the current cloud and the static map 
    icp_result = o3d.pipelines.registration.registration_icp(
            input_cloud_static_downsampled, static_map, icp_params["threshold"], np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria = icp_params["criteria"]
        )
    
    # Update the current transformation for the next iteration
    current_transformation = np.dot(current_transformation, icp_result.transformation)
    # transformations.append(icp_result.transformation)

    # Apply ICP transformation to the current cloud and person cloud
    input_cloud_static_downsampled.transform(icp_result.transformation)
    person_cloud.transform(icp_result.transformation)
    trajectory_cloud.transform(icp_result.transformation)
    
    # Trajectory update
    trajectory_cloud.transform(icp_result.transformation)
    change_colors(trajectory_cloud, COLORS["RED"])
    trajectory_global_cloud += trajectory_cloud
    
    # Filter and merge into static map
    filtered_cloud = filter_close_points(static_map, input_cloud_static_downsampled, CONFIG["radius"])
    static_map, max_count = add_to_static_map(static_map, filtered_cloud, icp_params["voxel_counter"], 
                                            icp_params["voxel_size"], max_count)
        
    # Update the person cloud and the global cloud
    person_global_cloud += person_cloud
    
    return predictions, filtered_cloud, trajectory_cloud, person_cloud, current_transformation, static_map, max_count
    # # return predictions, {
    #     "static": current_static_cloud_aligned,
    #     "trajectory": current_trajectory_aligned,
    #     "person": current_person_aligned
    # }

if __name__ == "__main__":
    clear_terminal()
    
    static_map1, trajectory_global_cloud1, person_global_cloud1, show_cloud1 = initialize_maps()
    static_map2, trajectory_global_cloud2, person_global_cloud2, show_cloud2 = initialize_maps()
    
    # CSV files for the two LiDAR captures to fuse
    # Note: Both paths are the same for now, but they can be different if needed
    folder_path1 = FILE_NAME[CONFIG["selected_file"]]
    folder_path2 = FILE_NAME[CONFIG["selected_file"]]
    csv_files1 = get_csv_files(folder_path1)
    csv_files2 = get_csv_files(folder_path2)
    
    # Initialize the Roboflow model
    model = initialize_roboflow()
    
    if CONFIG["print_realtime"]:
        vis_map = setup_visualizer()
    else:
        vis_map = None
    
    icp_params = icp_initialize()
    max_count = icp_params["max_count"]
    current_transformation = np.eye(4)
    
    for i in range(0, min(len(csv_files1), CONFIG["max_number_of_clouds"])):
    # for i in range(0, 50):
        print("Starting iteration:", i)
        
        input_cloud1 = load_csv_as_open3d_point_cloud(csv_files1[i], folder_path1)
        predictions1, current_static_cloud_aligned1, current_trajectory_aligned1, current_person_aligned1, current_transformation, static_map1, max_count = icp_alignment(csv_files1, folder_path1, i, input_cloud1, static_map1, trajectory_global_cloud1, person_global_cloud1, icp_params, current_transformation, max_count)
        
        print(len(static_map1.points), "points in static map after alignment")
        # predictions1, maps1 = icp_alignment(...) TODO: change return types
        # static1 = maps1["static"]
        # trajectory1 = maps1["trajectory"]
        # person1 = maps1["person"]
        show_cloud1 += current_person_aligned1 + current_static_cloud_aligned1 + current_trajectory_aligned1
        
        if CONFIG["print_realtime"]:
            display_point_cloud(vis_map, show_cloud1, point_size=1.0)
        
        # input_cloud2.transform
    if CONFIG["print_realtime"]:
        vis_map.run()