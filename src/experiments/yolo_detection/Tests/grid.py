import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from roboflow import Roboflow
import open3d as o3d
import pygetwindow as gw


def create_xy_grid(x_range=(-5, 5), y_range=(-5, 5), step=1.0, z_level=0.0):
    lines = []
    points = []

    # Vertical lines (constant x, vary y)
    for x in np.arange(x_range[0], x_range[1] + step, step):
        points.append([x, y_range[0], z_level])
        points.append([x, y_range[1], z_level])
        lines.append([len(points) - 2, len(points) - 1])

    # Horizontal lines (constant y, vary x)
    for y in np.arange(y_range[0], y_range[1] + step, step):
        points.append([x_range[0], y, z_level])
        points.append([x_range[1], y, z_level])
        lines.append([len(points) - 2, len(points) - 1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for _ in lines])  # gray grid lines
    return line_set

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')    




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
    grid = create_xy_grid(x_range=(-10, 10), y_range=(-10, 10), step=1.0)
    vis.add_geometry([point_cloud, grid])
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    
    #time.sleep(1)
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    #vis.run()        

def show_csv(file_path):
    pc = pd.read_csv(file_path, usecols=["x(m)", "y(m)", "z(m)"])
    # Converts a DataFrame into an Open3D PointCloud object
    x, y, z = pc["x(m)"], pc["y(m)"], pc["z(m)"]
    points = np.vstack((x, y, z)).T
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points) 
    #print(np.asarray(pc.points))
    
    vis = setup_visualizer()
    display_point_cloud(vis, pc, 2.0)
    vis.run()



if __name__ == "__main__":
    clear_terminal()
    # Convert initial CSV to PNG
    # Show CSV
    #show_csv("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/frame_083.csv")
    # Detect objects from PNG with YOLO
    # Remove points from CSV
    show_csv("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/solo_test/solo_person.csv")