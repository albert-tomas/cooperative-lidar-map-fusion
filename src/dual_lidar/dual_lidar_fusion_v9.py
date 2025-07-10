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
from scipy.spatial import cKDTree


# This script will perform the fusion of two LiDAR captures using ICP and YOLO for object detection, this version for now only uses one LiDAR capture to generate a single map
# v2 - Added grey scale to the map, added option to save the maps
# v3 - Changed parameters and return variables to icp_alignment function, for better clarity and functionality
# v4 - Added alignment for the second LiDAR capture
# v5 - Cleaned up the code, adding modularity and better structure for the main function
# v6 - Continued v5
# v7 - Applying the initial transformation was not working properly, as it was messing with the coordinates of the point clouds and the coordinates of the boxes, so we created a new program that applies a transformation (translation and rotation) to all the CSV in the file, this way we are not simulating a different capture, but we actually have two different captures
# v7 - Added visualization of the fused map (for now it is just the first map plus the second map) when the predictions for both LiDAR captures are available
# v8 - Created a function to transform the map of the second LiDAR to align it with the first LiDAR map using the person detection predictions, this way the maps are already aligned in position, but not orientation
# v9 - Added function to calculate the best rotation to align the both maps

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
    "print_realtime": False,
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

def icp_alignment(csv_files, folder_path, i, input_cloud, maps, icp_params, current_transformation, max_count, model, lidar_name):

    # Initialize global clouds
    static_map = maps["static"]
    trajectory_global_cloud = maps["trajectory"]
    person_global_cloud = maps["person"]

    # Load cloud and create BEV + predictions
    current_csv = csv_files[i]
    bev_image_path = os.path.join(icp_params["base_dir"] + "_" + lidar_name, f"image_{i}.png")
    create_bev_image(
        os.path.join(folder_path, current_csv), bev_image_path,
        res=BEV_CONFIG["res"], x_range=BEV_CONFIG["x_range"], y_range=BEV_CONFIG["y_range"], z_range=BEV_CONFIG["z_range"]
    )
    predictions = yolo_add_boxes(bev_image_path, 
                                os.path.join(icp_params["base_dir"] + "_" + lidar_name, f"image_{i}_boxes.png"), model)
    
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
        trajectory_global_cloud += trajectory_cloud
        
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

def initialize_lidar(lidar_name):
    static_map, trajectory, person, show_cloud = initialize_maps()
    maps = {
        "static": static_map,
        "trajectory": trajectory,
        "person": person
    }
    if lidar_name == "lidar1":
        folder_path = FILE_NAME[CONFIG["selected_file"]]
    else:
        folder_path = FILE_NAME[CONFIG["selected_file"]] + "_transformed"
        
    csv_files = get_csv_files(folder_path)
    icp_params = icp_initialize()
    vis_map = setup_visualizer() if CONFIG["print_realtime"] else None
    return {
        "maps": maps,
        "folder_path": folder_path,
        "csv_files": csv_files,
        "icp_params": icp_params,
        "current_transformation": np.eye(4),
        "max_count": icp_params["max_count"],
        "vis_map": vis_map,
        "show_cloud": show_cloud,
        "name": lidar_name
    }


def process_frame(i, lidar, model):
    
    input_cloud = load_csv_as_open3d_point_cloud(lidar["csv_files"][i], lidar["folder_path"])
    
    predictions, results = icp_alignment(
        lidar["csv_files"], lidar["folder_path"], i,
        input_cloud, lidar["maps"], lidar["icp_params"],
        lidar["current_transformation"], lidar["max_count"], model, lidar["name"]
    )
    
    lidar["current_transformation"] = results["current_transformation"]
    lidar["max_count"] = results["max_count"]
    lidar["maps"]["static"]     = results["static_global"]
    lidar["maps"]["trajectory"] = results["trajectory_global"]
    lidar["maps"]["person"]     = results["person_global"]
    
    show_cloud = lidar["maps"]["static"] + lidar["maps"]["trajectory"]
    if CONFIG["realtime_people_only"]:
        show_cloud += results["current_person_aligned"]
    else:
        show_cloud += lidar["maps"]["person"]
    
    lidar["show_cloud"] = show_cloud
    
    return predictions, lidar

def save_all_maps(lidar, timestamp, base_path):
    timestamp = datetime.now().strftime("%d%m_%H%M")  # Format: daymonth_hourminute
    base_path = f"C:/Users/Albert/Desktop/lidar-fused-3d-map/src/visualization/saved_maps_and_trajectories/{CONFIG['selected_file']}"
    
    save_static_map(lidar["maps"]["static"], timestamp, base_path)
    save_trajectory(lidar["maps"]["trajectory"], timestamp, base_path)
    if len(lidar["maps"]["person"].points) <= 0:
        lidar["maps"]["person"] = generate_single_point()
        change_colors(lidar["maps"]["person"], COLORS["GREY"])
    save_person_cloud(lidar["maps"]["person"], timestamp, base_path)

def initialize_fusion_viewer(show_cloud1, show_cloud2):
    # combined = o3d.geometry.PointCloud()
    # combined += show_cloud1
    # combined += show_cloud2
    fused_cloud = show_cloud1 + show_cloud2
    return fused_cloud

def check_both_predictions(pred1, pred2):
    # print(f"predictions for LiDAR 1: {pred1}")
    # print(f"predictions for LiDAR 2: {pred2}")
    if len(pred1["predictions"]) >= 1 and len(pred2["predictions"]) >= 1:
        # print("Both LiDAR captures have predictions.")
        return True
    return False

def get_center_world_coords(pred, scale_x, scale_y, bev_height, res=0.02, x_range=(-5, 5), y_range=(-5, 5)):
        # Center coordinates in pixels
        cx_px = pred["x"] * scale_x
        cy_px = pred["y"] * scale_y

        # Invert y coordinate and transform to meters
        x_m = x_range[0] + cx_px * res
        y_idx = bev_height - cy_px
        y_m = y_range[0] + y_idx * res

        return np.array([x_m, y_m])

def compute_overlap_score(cloud_ref, test_cloud, threshold=0.1):
    
    ref_points = np.asarray(cloud_ref.points)
    test_points = np.asarray(test_cloud.points)

    if len(ref_points) == 0 or len(test_points) == 0:
        return 0.0 

    tree = cKDTree(ref_points)
    distances, _ = tree.query(test_points, k=1)

    overlap_count = np.sum(distances < threshold)
    score = overlap_count / len(test_points)

    return score

def calculate_best_rotation(cloud_ref, cloud_to_align, angle_step_deg=6):
    
    best_score = -np.inf
    best_rotation = np.eye(4)

    vis_test = setup_visualizer()

    for angle in range(0, 360, angle_step_deg):
        theta = np.radians(angle)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta),  np.cos(theta), 0, 0],
            [0,              0,             1, 0],
            [0,              0,             0, 1]
        ])

        test_cloud = o3d.geometry.PointCloud()
        test_cloud.points = cloud_to_align.points
        test_cloud.colors = cloud_to_align.colors
        test_cloud.transform(R)

        score = compute_overlap_score(cloud_ref, test_cloud)
        print(f"Angle: {angle} degrees, Overlap Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_rotation = R
            best_angle = angle
        
        disp_cloud = cloud_ref + test_cloud
        
        display_point_cloud(vis_test, disp_cloud, point_size=1.5)
        time.sleep(1)
        if angle == 180:
            vis_test.run()

    
    print(f"Best rotation found at angle: {best_angle} degrees with score: {best_score}")

    return best_rotation

def align_map_by_person_detection(pred1, pred2, show_cloud1, show_cloud2,
                                res=0.02, x_range=(-5, 5), y_range=(-5, 5)):
    
    # BEV configuration
    bev_width = int((x_range[1] - x_range[0]) / res)
    bev_height = int((y_range[1] - y_range[0]) / res)
    yolo_width, yolo_height = 600, 600
    scale_x = bev_width / yolo_width
    scale_y = bev_height / yolo_height

    center1 = get_center_world_coords(pred1["predictions"][0], scale_x, scale_y, bev_height, res=0.02, 
                                    x_range=(-5, 5),     y_range=(-5, 5))
    center2 = get_center_world_coords(pred2["predictions"][0], scale_x, scale_y, bev_height, res=0.02, 
                                    x_range=(-5, 5), y_range=(-5, 5))
    translation = center1 - center2

    # Transform the second cloud
    translation_transform = np.eye(4)
    translation_transform[0, 3] = translation[0]
    translation_transform[1, 3] = translation[1]
    
    cloud2_translated = o3d.geometry.PointCloud()
    cloud2_translated += show_cloud2
    cloud2_translated.transform(translation_transform)
    
    rotation = calculate_best_rotation(show_cloud1, cloud2_translated)

    full_transform = np.dot(translation_transform, rotation)

    return full_transform

if __name__ == "__main__":
    clear_terminal()
    
    # Initialize the Roboflow model
    model = initialize_roboflow()
    
    lidar1 = initialize_lidar("lidar1")
    lidar2 = initialize_lidar("lidar2")
    
    maps_are_aligned = False
    
    for i in range(0, min(len(lidar1["csv_files"]), len(lidar2["csv_files"]), CONFIG["max_number_of_clouds"])):
        
        print("Starting iteration: ", i)
        
        pred1, lidar1 = process_frame(i, lidar1, model)
        # We will apply the initial transformation to the second LiDAR capture to simulate a different capture
        pred2, lidar2 = process_frame(i, lidar2, model)
        
        if check_both_predictions(pred1, pred2) and maps_are_aligned == False:
            print(f"Predictions found for frame {i}, starting map fusion.")
            vis_fusion = setup_visualizer() if CONFIG["print_realtime"] else None
            # We will align the second LiDAR map by the first LiDAR map using the person detection predictions
            lidar2_map_transformation = align_map_by_person_detection(pred1, pred2, lidar1["show_cloud"], lidar2["show_cloud"])
            maps_are_aligned = True

        if CONFIG["print_realtime"]:
            display_point_cloud(lidar1["vis_map"], lidar1["show_cloud"], point_size=1.5)
            display_point_cloud(lidar2["vis_map"], lidar2["show_cloud"], point_size=1.5)
            
            print(f"Number of points in LiDAR 1: {len(lidar1['show_cloud'].points)}")
            print(f"Number of points in LiDAR 2: {len(lidar2['show_cloud'].points)}")
            
        if maps_are_aligned == True:
            lidar2_map_transformed = o3d.geometry.PointCloud()
            lidar2_map_transformed += lidar2["show_cloud"]
            lidar2_map_transformed.transform(lidar2_map_transformation)
            fused_cloud = lidar1["show_cloud"] + lidar2_map_transformed
            
            if CONFIG["print_realtime"]:
                display_point_cloud(vis_fusion, fused_cloud, point_size=1.5)
                print(f"Number of points in the fused map: {len(fused_cloud.points)}")
    
    if CONFIG["save_map"]:
        # Since currently we only use one LiDAR capture to generate a single map, we will save the first map
        save_all_maps(lidar1)  
    
    if CONFIG["print_realtime"]:
        lidar1["vis_map"].run()
        lidar2["vis_map"].run()
        if maps_are_aligned:
            vis_fusion.run()