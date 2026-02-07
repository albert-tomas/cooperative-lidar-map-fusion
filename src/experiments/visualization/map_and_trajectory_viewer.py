import open3d as o3d
import os
import pygetwindow as gw

OPTIONS = {
    "capture_name": "strada2", # Options: "catture", "strada1", "strada2", "strada3"
    "timestamp": "1505_1522", # Select the timestamp of the capture
    "print_map": True, # Print the map
    "print_trajectory": True, # Print the trajectory
}

def load_map(folder_path):
    if os.path.exists(folder_path):
        map_pcd = o3d.io.read_point_cloud(folder_path)
        print(f"Global map loaded from {folder_path}")
        return map_pcd
    else:
        print(f"No map file found at {folder_path}")
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

if __name__ == "__main__":
    # Base directory
    base_dir = "C:/Users/Albert/Desktop/lidar-fused-3d-map/src/visualization/saved_maps_and_trajectories"
    print_cloud = o3d.geometry.PointCloud()
    
    if OPTIONS["print_map"]:
        folder_path_map  = base_dir + "/" + OPTIONS["capture_name"] + "/map_" + OPTIONS["timestamp"] + ".ply"
        print(folder_path_map)
        map_pcd = load_map(folder_path_map)
        print_cloud += map_pcd
    if OPTIONS["print_trajectory"]:
        folder_path_trajectory  = base_dir + "/" + OPTIONS["capture_name"] + "/trajectory_" + OPTIONS["timestamp"] + ".ply"
        trajectory_pcd = load_map(folder_path_trajectory)
        print_cloud += trajectory_pcd

    vis = setup_visualizer()
    display_point_cloud(vis, print_cloud)
    vis.run()