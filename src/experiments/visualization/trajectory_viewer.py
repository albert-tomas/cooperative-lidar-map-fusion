import numpy as np
import open3d as o3d
import os
import pygetwindow as gw

# def load_transformations(file_name):
#     folder_path = "C:/Users/Albert/Desktop/lidar-fused-3d-map/src/visualization/saved_transformations"
#     file_path = os.path.join(folder_path, file_name)
#     if os.path.exists(file_path):
#         transformations = np.load(file_path, allow_pickle=True)
#         print(f"Transformations loaded from {file_path}")
#         return transformations
#     else:
#         print(f"No transformations file found at {file_path}")
#         return None

# def setup_visualizer():
#     # Creates and configures an Open3D visualizer
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     # Get the window and maximize it
#     window = gw.getWindowsWithTitle('Open3D')[0]
#     window.maximize()
#     return vis

# def display_point_cloud(vis, point_cloud, point_size=5.0):
#     # Displays the point cloud in the visualizer
#     vis.clear_geometries()
#     vis.add_geometry(point_cloud)
#     render_option = vis.get_render_option()
#     render_option.point_size = point_size
    
#     #time.sleep(1)
#     vis.update_geometry(point_cloud)
#     vis.poll_events()
#     vis.update_renderer()
#     #vis.run()

# def generate_single_point():
#     # Generate a single point at (0, 0, 0)
#     points = np.array([[0.0, 0.0, 0.0]])

#     # Create Open3D point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)

#     return pcd

# def visualize_trajectory(transformations):
#     transformed_cloud = generate_single_point()
#     combined_cloud = o3d.geometry.PointCloud()
    
#     for i, transform in enumerate(transformations):
#         print(f"Applying transformation {i+1}/{len(transformations)}")
#         transformed_cloud.transform(transform)
#         combined_cloud += transformed_cloud  # Adds only once per iteration
    
#     return combined_cloud

# if __name__ == "__main__":
#     file_name = "transformations.npy"
#     transformations = load_transformations(file_name)
#     if transformations is not None:
#         combined_cloud = visualize_trajectory(transformations)
#         vis = setup_visualizer()
#         display_point_cloud(vis, combined_cloud)
#         vis.run()
#     else:
#         print("No transformations to visualize.")

def load_map():
    folder_path = "C:/Users/Albert/Desktop/lidar-fused-3d-map/src/visualization/saved_trajectories"
    file_path = os.path.join(folder_path, "global_trajectory.ply")
    if os.path.exists(file_path):
        trajectory_pcd = o3d.io.read_point_cloud(file_path)
        print(f"Global map loaded from {file_path}")
        return trajectory_pcd
    else:
        print(f"No map file found at {file_path}")
        return None

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

if __name__ == "__main__":
    trajectory_pcd = load_map()
    print(trajectory_pcd)
    if trajectory_pcd is not None:
        vis = setup_visualizer()
        display_point_cloud(vis, trajectory_pcd)
        vis.run()
    else:
        print("No point cloud to display.")