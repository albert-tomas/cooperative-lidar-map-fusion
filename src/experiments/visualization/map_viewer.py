import open3d as o3d
import os
import pygetwindow as gw

def load_map():
    folder_path = "C:/Users/Albert/Desktop/lidar-fused-3d-map/src/visualization/saved_maps"
    file_path = os.path.join(folder_path, "global_map.ply")
    if os.path.exists(file_path):
        map_pcd = o3d.io.read_point_cloud(file_path)
        print(f"Global map loaded from {file_path}")
        return map_pcd
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
    map_pcd = load_map()
    if map_pcd is not None:
        vis = setup_visualizer()
        display_point_cloud(vis, map_pcd)
        vis.run()
    else:
        print("No point cloud to display.")