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
# v2 - Added grey scale to the map, added option to save the maps
# v3 - Changed parameters and return variables to icp_alignment function, for better clarity and functionality
# v4 - Added alignment for the second LiDAR capture

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

def icp_alignment(csv_files, folder_path, i, input_cloud, maps, icp_params, current_transformation, max_count):

    # Initialize global clouds
    static_map = maps["static"]
    trajectory_global_cloud = maps["trajectory"]
    person_global_cloud = maps["person"]

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
        static_map += input_cloud_static_downsampled
        
        change_colors(trajectory_cloud, COLORS["BLUE"])
        trajectory_global_cloud = trajectory_cloud
        
        results = {
            "current_static_cloud_aligned": input_cloud_static_downsampled,
            "current_trajectory_aligned": trajectory_cloud,
            "current_person_aligned": person_cloud,
            "current_transformation": current_transformation,
            "static_global": static_map,
            "trajectory_global": trajectory_global_cloud,
            "person_global": person_global_cloud,
            "max_count": max_count
        }
        
        return predictions, results
    
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
    
    results = {
            "current_static_cloud_aligned": filtered_cloud,
            "current_trajectory_aligned": trajectory_cloud,
            "current_person_aligned": person_cloud,
            "current_transformation": current_transformation,
            "static_global": static_map,
            "trajectory_global": trajectory_global_cloud,
            "person_global": person_global_cloud,
            "max_count": max_count
        }
    
    return predictions, results

if __name__ == "__main__":
    clear_terminal()
    
    static_map1, trajectory_global_cloud1, person_global_cloud1, show_cloud1 = initialize_maps()
    maps1 = {
        "static": static_map1,
        "trajectory": trajectory_global_cloud1,
        "person": person_global_cloud1
    }
    static_map2, trajectory_global_cloud2, person_global_cloud2, show_cloud2 = initialize_maps()
    maps2 = {
        "static": static_map2,
        "trajectory": trajectory_global_cloud2,
        "person": person_global_cloud2
    }
    
    # CSV files for the two LiDAR captures to fuse
    # Note: Both paths are the same for now, but they can be different if needed
    folder_path1 = FILE_NAME[CONFIG["selected_file"]]
    folder_path2 = FILE_NAME[CONFIG["selected_file"]]
    csv_files1 = get_csv_files(folder_path1)
    csv_files2 = get_csv_files(folder_path2)
    
    # Initialize the Roboflow model
    model = initialize_roboflow()
    
    if CONFIG["print_realtime"]:
        vis_map1 = setup_visualizer()
        vis_map2 = setup_visualizer()
    else:
        vis_map1 = None
        vis_map2 = None
    
    icp_params1 = icp_initialize()
    icp_params2 = icp_initialize()

    max_count1 = icp_params1["max_count"]
    current_transformation1 = np.eye(4)
    max_count2 = icp_params2["max_count"]
    current_transformation2 = np.eye(4)
    
    for i in range(0, min(len(csv_files1), len(csv_files2), CONFIG["max_number_of_clouds"])):
        print("Starting iteration:", i)
        
        input_cloud1 = load_csv_as_open3d_point_cloud(csv_files1[i], folder_path1)
        predictions1, results1 = icp_alignment(csv_files1, folder_path1, i, input_cloud1, maps1, icp_params1, current_transformation1, max_count1)
        
        current_transformation1 = results1["current_transformation"]
        max_count1 = results1["max_count"]
        maps1["static"]     = results1["static_global"]
        maps1["trajectory"] = results1["trajectory_global"]
        maps1["person"]     = results1["person_global"]
        
        show_cloud1 = static_map1 + trajectory_global_cloud1
        
        input_cloud2 = load_csv_as_open3d_point_cloud(csv_files1[i], folder_path2)
        predictions2, results2 = icp_alignment(csv_files2, folder_path2, i, input_cloud2, maps2, icp_params2, current_transformation2, max_count2)
        
        current_transformation2 = results2["current_transformation"]
        max_count2 = results2["max_count"]
        maps2["static"]     = results2["static_global"]
        maps2["trajectory"] = results2["trajectory_global"]
        maps2["person"]     = results2["person_global"]
        
        show_cloud2 = static_map2 + trajectory_global_cloud2
        
        if CONFIG["realtime_people_only"]:
            show_cloud1 += results1["current_person_aligned"]
            show_cloud2 += results2["current_person_aligned"]
        else:
            show_cloud1 += person_global_cloud1
            show_cloud2 += person_global_cloud2
        
        if CONFIG["print_realtime"]:
            display_point_cloud(vis_map1, show_cloud1, point_size=1.5)
            display_point_cloud(vis_map2, show_cloud2, point_size=1.5)
        
        # input_cloud2.transform
        
    if CONFIG["save_map"]:
        # Since currently we only use one LiDAR capture to generate a single map, we will save the first map
        timestamp = datetime.now().strftime("%d%m_%H%M")  # Format: daymonth_hourminute
        base_path = f"C:/Users/Albert/Desktop/lidar-fused-3d-map/src/visualization/saved_maps_and_trajectories/{CONFIG['selected_file']}"
        save_static_map(static_map1, timestamp, base_path)
        save_trajectory(trajectory_global_cloud1, timestamp, base_path)
        if len(person_global_cloud1.points) <= 0:
            person_global_cloud1 = generate_single_point()
            change_colors(person_global_cloud1, COLORS["GREY"])
        save_person_cloud(person_global_cloud1, timestamp, base_path)
        
    if CONFIG["print_realtime"]:
        vis_map1.run()
        vis_map2.run()