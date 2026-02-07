import os
import pandas as pd
import open3d as o3d
import numpy as np
import time
import pygetwindow as gw


# Configuration
CONFIG = {
    "max_number_of_clouds": 250,
    "voxel_size": 0.5,
    "print_realtime": True,
    "folder_path_csv": "C:/Users/Albert/Desktop/lidar-fused-3d-map/CSVfiles",
    "save_map": True,
    "save_transformations": True
}

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_csv_files(folder_path):
    # Returns a list of CSV files in the given directory
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    # Check if there are any CSV files 
    if not csv_files:
        print("No CSV files found in the folder.")
    else:
        print(f"Found {len(csv_files)} CSV files.")
    return csv_files

def setup_visualizer():
    # Creates and configures an Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Get the window and maximize it
    window = gw.getWindowsWithTitle('Open3D')[0]
    window.maximize()
    return vis

def display_point_cloud(vis, point_cloud, point_size=2.0):
    # Displays the point cloud in the visualizer
    vis.clear_geometries()
    vis.add_geometry(point_cloud)
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    
    #time.sleep(1)
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    #vis.run()

def load_csv_as_open3d_point_cloud(csv_file, folder_path):
    full_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(full_path, usecols=["x(m)", "y(m)", "z(m)"])
    points = np.vstack((df["x(m)"], df["y(m)"], df["z(m)"])).T

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    return pc

def filter_close_points(global_map, new_cloud, radius):
    # Create a KDTree for the existing global map
    kdtree = o3d.geometry.KDTreeFlann(global_map)
    
    # Prepare a list for the filtered points
    filtered_points = []

    for point in new_cloud.points:
        [_, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        if len(idx) == 0:  # No nearby points within the radius
            filtered_points.append(point)

    # Create a new point cloud with only the filtered points
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    return filtered_cloud

def icp_alignment(csv_files, folder_path, vis, max_number_of_clouds, voxel_size=1.0):
    threshold = 0.5  # Distance threshold for ICP matching. Correspondance distance
    max_iterations = 50 # Maximum number of iterations for ICP
    
    # Step 1: Create the initial global map (reference cloud)
    input_cloud = load_csv_as_open3d_point_cloud(csv_files[0], folder_path)
    global_map = input_cloud.voxel_down_sample(voxel_size)

    # Display the initial point cloud
    if CONFIG["print_realtime"]:
        display_point_cloud(vis, global_map, point_size=2.0)
    
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iterations)
    
    # Initialize an empty list to store transformations
    transformations = []    
    # Initial transformation (identity matrix)
    current_transformation = np.eye(4)

    
    # Iterate over the next point clouds
    for i in range(1, min(len(csv_files), max_number_of_clouds)):
        current_cloud = load_csv_as_open3d_point_cloud(csv_files[i], folder_path)
        current_cloud_downsampled = current_cloud.voxel_down_sample(voxel_size)
        
        #global_map += current_cloud
        #display_point_cloud(vis, global_map, point_size=2.0)
        
        # Apply the accumulated transformation to the local cloud
        current_cloud_downsampled.transform(current_transformation)
        
        # Run ICP between the current cloud and the global map 
        icp_result = o3d.pipelines.registration.registration_icp(
            current_cloud_downsampled, global_map, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria = criteria
        )
        
        current_transformation = np.dot(current_transformation, icp_result.transformation)

        # Transform the current cloud and add it to the global map
        current_cloud_downsampled.transform(icp_result.transformation)
        # Store the transformation
        transformations.append(icp_result.transformation)
        
        # Filter points that are too close to existing points in the global map
        filtered_cloud = filter_close_points(global_map, current_cloud_downsampled, radius=0.1)

        # Add the transformed cloud to the global map
        global_map.points.extend(filtered_cloud.points)
        #global_map = global_map.voxel_down_sample(voxel_size)
        
        print(f"Cloud {i} aligned. Total number of points in global map: {len(global_map.points)}")
        if CONFIG["print_realtime"]:
            display_point_cloud(vis, global_map, point_size=2.0)
    
    return global_map, transformations

def save_map(global_map):
    # Save the global map to a file
    folder_path_save_map = "C:/Users/Albert/Desktop/lidar-fused-3d-map/src/visualization/saved_maps"
    output_path = os.path.join(folder_path_save_map, "global_map.ply")
    o3d.io.write_point_cloud(output_path, global_map)
    print(f"Global map saved to {output_path}")

def save_transformations(transformations):
    folder_path = "C:/Users/Albert/Desktop/lidar-fused-3d-map/src/visualization/saved_transformations"
    output_path = os.path.join(folder_path, "transformations.npy")
    np.save(output_path, transformations)
    print(f"Transformations saved to {output_path}")

def main():
    start_time = time.time()
    
    # Variables
    max_number_of_clouds = CONFIG["max_number_of_clouds"]
    voxel_size = CONFIG["voxel_size"]
    
    # Folder path with all the CSV files
    #folder_path = "C:/Users/Albert/Desktop/lidar-fused-3d-map/CSVfiles"
    folder_path = CONFIG["folder_path_csv"]
    # folder_path = "D:/lidar-thesis/PCAP_CSV_files/Capture1911/CSV"
    csv_files = get_csv_files(folder_path)
    
    # Initialize and run the visualizer
    if CONFIG["print_realtime"]:
        vis = setup_visualizer()
    else:
        vis = None
    
    global_map, transformations = icp_alignment(csv_files, folder_path, vis, max_number_of_clouds, voxel_size)
    
    if CONFIG["save_map"]:
        save_map(global_map)
        
    if CONFIG["save_transformations"]:
        save_transformations(transformations)
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    if CONFIG["print_realtime"]:
        vis.run()    
    


if __name__ == "__main__":
    clear_terminal()
    main()