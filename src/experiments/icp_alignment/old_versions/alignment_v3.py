import os
import pandas as pd
import open3d as o3d
import numpy as np
import time
import pygetwindow as gw

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

def load_point_clouds(csv_files, folder_path, max_number_of_clouds=10):
    # Loads up to max_iterations point clouds from CSV files
    # Each point cloud has the x, y, z coordinates and the file name
    point_clouds = []
    for i, file in enumerate(csv_files):
        if i >= max_number_of_clouds:
            break
        full_path = os.path.join(folder_path, file)
        pc = pd.read_csv(full_path, usecols=["x(m)", "y(m)", "z(m)"])
        pc["file"] = file
        point_clouds.append(pc)
        print(i)
    return point_clouds

def create_open3d_point_cloud(df):
    # Converts a DataFrame into an Open3D PointCloud object
    x, y, z = df["x(m)"], df["y(m)"], df["z(m)"]
    points = np.vstack((x, y, z)).T
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    # TODO: maybe return the points as well if they have to be processed after, avoiding recalculating
    return pc

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
    
    time.sleep(1)
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    #vis.run()

def filter_by_z_threshold(point_cloud, vis):
    # Filters points by Z threshold and updates the visualizer
    
    # Get numpy array from Open3D point cloud
    points = np.asarray(point_cloud.points)
    
    # Compute Z threshold
    z = points[:, 2]
    z_min = np.min(z)
    z_max = np.max(z)
    z_threshold = z_min + (z_max - z_min) / 2
    
    filtered_points = points[z < z_threshold]
    point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    
    time.sleep(1)
    vis.clear_geometries()
    vis.add_geometry(point_cloud)
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    #time.sleep(1)    

def icp_alignment(point_clouds, vis, max_number_of_clouds=10, voxel_size=1.0):
    
    # Step 1: Create the initial global map (reference cloud)
    input_cloud = create_open3d_point_cloud(point_clouds[0])
    global_map = input_cloud.voxel_down_sample(voxel_size)

    # Display the initial point cloud
    display_point_cloud(vis, global_map, point_size=2.0)

    
    # Iterate over the next point clouds
    for i in range(1, min(len(point_clouds), max_number_of_clouds)):
        current_cloud = create_open3d_point_cloud(point_clouds[i])
        current_cloud = current_cloud.voxel_down_sample(voxel_size)
        
        global_map += current_cloud
        display_point_cloud(vis, global_map, point_size=2.0)
        
        # Run ICP between the current cloud and the global map
        threshold = 0.5  # Distance threshold for ICP matching. Correspondance distance
        icp_result = o3d.pipelines.registration.registration_icp(
            current_cloud, global_map, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        # Transform the current cloud and add it to the global map
        transformed_cloud = current_cloud.transform(icp_result.transformation)
        global_map += transformed_cloud
        global_map = global_map.voxel_down_sample(voxel_size)
        
        print(f"Cloud {i} aligned.")
        display_point_cloud(vis, global_map, point_size=2.0)
    
    

def main():
    # Variables
    max_number_of_clouds = 100
    voxel_size = 0.5
    
    # Load all CSV files from a folder
    folder_path = "D:/lidar-thesis/Capture1911/CSV"
    csv_files = get_csv_files(folder_path)

    # Load point clouds from CSV files
    point_clouds = load_point_clouds(csv_files, folder_path, max_number_of_clouds)

    # # Create Open3D point cloud from the first CSV
    # first_pc = point_clouds[0]
    # point_cloud = create_open3d_point_cloud(first_pc)

    # # Initialize and run the visualizer
    vis = setup_visualizer()
    # display_point_cloud(vis, point_cloud, point_size=2.0)   

    # # Filter by Z and update visualization
    # filter_by_z_threshold(point_cloud, vis)
    
    icp_alignment(point_clouds, vis, max_number_of_clouds, voxel_size)
    
    vis.run()


if __name__ == "__main__":
    clear_terminal()
    main()