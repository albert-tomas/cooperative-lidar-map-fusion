import os
import pandas as pd
import open3d as o3d
import numpy as np
import time
import pygetwindow as gw

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

def apply_colors(point_cloud, threshold=2, above_color=[0, 0, 1], below_color=[1, 0, 0]):
    # Assign colors based on the threshold
    z_values = np.asarray(point_cloud.points)[:, 2]
    colors = np.zeros((len(z_values), 3))
    colors[z_values > threshold] = above_color
    colors[z_values <= threshold] = below_color
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

def change_colors(point_cloud, color):
    num_points = len(point_cloud.points)
    colors = np.tile(color, (num_points, 1))
    point_cloud.colors = o3d.utility.Vector3dVector(colors)



def main():
    start_time = time.time()
    
    # Variables
    
    
    # Folder path with all the CSV files
    #folder_path = "C:/Users/Albert/Desktop/lidar-fused-3d-map/CSVfiles"
    folder_path = "C:/Users/Albert/Desktop/lidar-fused-3d-map/CSVfiles"
    # folder_path = "D:/lidar-thesis/PCAP_CSV_files/Capture1911/CSV"
    #csv_files = get_csv_files(folder_path)
    point_cloud = load_csv_as_open3d_point_cloud("points_info_20200520_000118_999.csv", folder_path)
    # Initialize and run the visualizer
    vis_map = setup_visualizer()
    
    point_cloud2 = load_csv_as_open3d_point_cloud("points_info_20200520_000118_999.csv", folder_path)
    
    
    transformation = np.array([
    [1, 0, 0, 0.5],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
    ])

    # Aplicar la transformación
    point_cloud2.transform(transformation)
    
    #apply_colors(point_cloud, threshold=1.0, above_color=[0.5, 0, 0], below_color=[255, 0, 0])
    change_colors(point_cloud, [0, 1, 0])
    change_colors(point_cloud2, [0, 0, 1])
    point_cloud += point_cloud2
    display_point_cloud(vis_map, point_cloud, point_size=2.0)
    
    #global_map, transformations = icp_alignment(csv_files, folder_path, vis_map, vis_trajectory)
    
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    vis_map.run()
    


if __name__ == "__main__":
    main()