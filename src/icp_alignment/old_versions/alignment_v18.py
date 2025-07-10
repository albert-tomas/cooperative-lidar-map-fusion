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


# v8 - Added a function to filter points that are too close to existing points in the global map
# v9 - Added a function to color the point clouds (red for trajectory and grey for the global map)
# v10 - Changed trajectory to be saved as a point cloud instead of a list of transformations
# v11 - Added object detection and removal
# v12 - Tried a new approach to object detection and removal compared to v11
# v13 - Added visualization of the 3 maps (the global map, the trajectory and the person cloud)
# v14 - Added sphere to the trajectory to visualize it better
# v15 - Cleaned up the code, added comments, and improved the structure
# v16 - Added noise removal
# ~~v17 - Discarded: Changed YOLO to be used locally to make it faster~~ (Discarded: worse results and actually slower)
# v18 - Added configuration parameters for the BEV image generation


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
    #"folder_path_csv": "C:/Users/Albert/Desktop/lidar-fused-3d-map/CSVfiles",
    "save_map": False,
    "remove_realtime": True, # If True, the detected objects will be removed in real-time from the past frames
    # "save_trajectory": False,
    "trajectory_sphere": True, # If True, the trajectory will be visualized as a sphere (for better visualization)
    "radius": 0.2, # Radius for filtering close points
}

BEV_CONFIG = {
    "res": 0.02,
    "x_range": (-5, 5),
    "y_range": (-5, 5),
    "z_range": (-2, 2),
    "scale": 2,  # Scale factor for the BEV image
}

COLORS = {
    "RED": [1, 0, 0],
    "GREEN": [0, 1, 0],
    "BLUE": [0, 0, 1],
    "GREY": [0.5, 0.5, 0.5],
}

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_csv_files(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the folder.")
    else:
        print(f"Found {len(csv_files)} CSV files.")
    return csv_files

def setup_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
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
    filtered_colors = []

    # Convert points and colors to list for indexing
    points = np.asarray(new_cloud.points)
    colors = np.asarray(new_cloud.colors) if new_cloud.has_colors() else None

    for i, point in enumerate(points):
        [_, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        if len(idx) == 0:  # No nearby points within the radius
            filtered_points.append(point)
            if colors is not None:
                filtered_colors.append(colors[i])

    # Create a new point cloud with only the filtered points
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)

    if filtered_colors:
        filtered_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
    return filtered_cloud

def generate_single_point():
    # Generate a single point at (0, 0, 0)
    points = np.array([[0.0, 0.0, 0.0]])

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def generate_sphere(radius=0.1, point_count=1000):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_cloud = sphere.sample_points_uniformly(number_of_points=point_count)

    return sphere_cloud

def change_colors(point_cloud, color):
    num_points = len(point_cloud.points)
    colors = np.tile(color, (num_points, 1))
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

def create_bev_image(csv_file, output_image_path, res=0.02, x_range=(-5, 5), y_range=(-5, 5), z_range=(-2, 2)):
    # Load data from CSV
    df = pd.read_csv(csv_file)

    # Define the grid indices
    x_bins = np.arange(x_range[0], x_range[1], res)
    y_bins = np.arange(y_range[0], y_range[1], res)

    # Histogram computation
    x_indices = np.digitize(df['x(m)'], x_bins) - 1
    y_indices = np.digitize(df['y(m)'], y_bins) - 1

    # Filter points within specified range
    valid_points = (x_indices >= 0) & (x_indices < len(x_bins)) & \
                (y_indices >= 0) & (y_indices < len(y_bins)) & \
                (df['z(m)'] >= z_range[0]) & (df['z(m)'] <= z_range[1])

    # Create an empty image
    bev_image = np.zeros((len(y_bins), len(x_bins)), dtype=np.uint8)

    # Accumulate points in the image array
    np.add.at(bev_image, (y_indices[valid_points], x_indices[valid_points]), 1)

    # Plot the image
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    bev_image = np.clip(bev_image * BEV_CONFIG["scale"], 0, 255) 
    ax.imshow(bev_image, cmap='gray', origin='lower')

    # Remove axes and padding
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Save the figure
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def initialize_roboflow():
    # Initialize Roboflow
    rf = Roboflow(api_key="m3mVIcAJXOFxaob0eVAt")

    # Load the trained model
    model = rf.workspace("sicariata").project("person_lidar").version(1).model
    return model

def yolo_add_boxes(image_file, output_image_path, model):
    
    # Load the input image using OpenCV
    image = cv2.imread(image_file)

    # Run the model prediction (Roboflow format)
    predictions = model.predict(image, confidence=10, overlap=30).json()

    # Draw bounding boxes and labels on the image
    for prediction in predictions['predictions']:
        # Calculate top-left and bottom-right coordinates
        x1 = int(prediction['x'] - prediction['width'] / 2)
        y1 = int(prediction['y'] - prediction['height'] / 2)
        x2 = int(prediction['x'] + prediction['width'] / 2)
        y2 = int(prediction['y'] + prediction['height'] / 2)

        # Draw the bounding box in green
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Compose and draw the label with confidence
        label = f"{prediction['class']} ({prediction['confidence']*100:.2f}%)"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image with the drawn detections
    cv2.imwrite(output_image_path, image)

    return predictions

def extract_points_in_boxes(point_cloud, predictions,
                            res=0.02, x_range=(-5, 5), y_range=(-5, 5)):
    # Convert the point cloud to numpy for easier indexing
    points = np.asarray(point_cloud.points)

    # Image scale factors
    bev_width = int((x_range[1] - x_range[0]) / res)
    bev_height = int((y_range[1] - y_range[0]) / res)
    yolo_width, yolo_height = 600, 600
    scale_x = bev_width / yolo_width
    scale_y = bev_height / yolo_height
    
    selected_indices = []

    for pred in predictions["predictions"]:
        # Convert bounding box from YOLO image to real-world coordinates
        x1_px = (pred["x"] - pred["width"] / 2) * scale_x
        x2_px = (pred["x"] + pred["width"] / 2) * scale_x
        y1_px = (pred["y"] - pred["height"] / 2) * scale_y
        y2_px = (pred["y"] + pred["height"] / 2) * scale_y

        y1_idx = bev_height - y2_px
        y2_idx = bev_height - y1_px

        x_min = x_range[0] + x1_px * res
        x_max = x_range[0] + x2_px * res
        y_min = y_range[0] + y1_idx * res
        y_max = y_range[0] + y2_idx * res

        # Find and color the points within the box
        in_box = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                (points[:, 1] >= y_min) & (points[:, 1] <= y_max))
        indices = np.where(in_box)[0]
        selected_indices.extend(indices)

    if not selected_indices:
        return o3d.geometry.PointCloud()

    new_cloud = point_cloud.select_by_index(selected_indices)
    
    return new_cloud

def print_color_counts(pcd, color_map, eps=1e-3):
    cols = np.asarray(pcd.colors)
    counts = {}
    for name, rgb in color_map.items():
        mask = np.all(np.abs(cols - rgb) < eps, axis=1)
        counts[name] = int(mask.sum())
    
    print(f"Frame: "
            f"\t\t{counts['GREY']} grey,"
            f"\t\t{counts['GREEN']} green,"
            f"\t\t{counts['BLUE']} blue,"
            f"\t\t{counts['RED']} red")

def remove_points_by_proximity(cloud, to_remove, radius=0.2):
    if len(to_remove.points) == 0:
        return cloud
    
    kdtree = o3d.geometry.KDTreeFlann(to_remove)
    mask = []

    for pt in cloud.points:
        [_, idx, _] = kdtree.search_radius_vector_3d(pt, radius)
        mask.append(len(idx) == 0)  # Keep point if no close neighbor found

    mask = np.array(mask)
    indices_to_keep = np.where(mask)[0]
    return cloud.select_by_index(indices_to_keep)

def icp_alignment(csv_files, folder_path, vis_map):
    # Configuration
    threshold = 0.5  # Distance threshold for ICP matching. Correspondance distance
    max_iterations = 50 # Maximum number of iterations for ICP
    max_number_of_clouds = CONFIG["max_number_of_clouds"]
    voxel_size = CONFIG["voxel_size"]
    sel = CONFIG["selected_file"]
    base_dir = f"D:/LiDAR-captures/{sel}/bev_images"
    
    # Initialize the Roboflow model
    model = initialize_roboflow()
    
    # Load first cloud and create BEV + predictions
    first_csv = csv_files[0]
    input_cloud = load_csv_as_open3d_point_cloud(first_csv, folder_path)
    first_bev = os.path.join(base_dir, "image_0.png")
    create_bev_image(
        os.path.join(folder_path, first_csv), first_bev,
        res=BEV_CONFIG["res"], x_range=BEV_CONFIG["x_range"], y_range=BEV_CONFIG["y_range"], z_range=BEV_CONFIG["z_range"]
    )
    predictions = yolo_add_boxes(first_bev, os.path.join(base_dir, "image_0_boxes.png"), model)
    
    # Extract person points and color them
    person_global_cloud = o3d.geometry.PointCloud()
    person_cloud = extract_points_in_boxes(input_cloud, predictions, res=0.02, x_range=(-5, 5), y_range=(-5, 5))
    change_colors(person_cloud, COLORS["GREEN"])
    person_global_cloud += person_cloud
    
    # Build initial static map (gray) and remove person points
    input_cloud_static = remove_points_by_proximity(input_cloud, person_cloud, radius=0.05)
    static_map = input_cloud_static.voxel_down_sample(voxel_size)
    static_map, _ = static_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    change_colors(static_map, COLORS["GREY"])
    
    # Initialize trajectory: either point or sphere
    trajectory_cloud = generate_single_point()
    if CONFIG["trajectory_sphere"]:
        trajectory_cloud = generate_sphere(radius=0.05, point_count=100)  
    change_colors(trajectory_cloud, COLORS["BLUE"])
    trajectory_global_cloud = o3d.geometry.PointCloud()
    trajectory_global_cloud += trajectory_cloud

    # Initialize the global map with the first cloud
    show_cloud = o3d.geometry.PointCloud()
    show_cloud += static_map + trajectory_cloud + person_cloud

    # Display the initial point cloud
    if CONFIG["print_realtime"]:
        display_point_cloud(vis_map, show_cloud, point_size=1.0)
    
    # ICP setup
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iterations)
    # Initialize an empty list to store transformations and add the identity matrix
    transformations = []
    current_transformation = np.eye(4)

    # Iterate over remaining clouds
    for i in range(1, min(len(csv_files), max_number_of_clouds)):
        # Load cloud and create BEV + predictions
        current_cloud = load_csv_as_open3d_point_cloud(csv_files[i], folder_path)
        csv = csv_files[i]
        bev = os.path.join(base_dir, f"image_{i}.png")
        create_bev_image(os.path.join(folder_path, csv), bev,
                        res=BEV_CONFIG["res"], x_range=BEV_CONFIG["x_range"], y_range=BEV_CONFIG["y_range"], z_range=BEV_CONFIG["z_range"])
        predictions = yolo_add_boxes(bev, os.path.join(base_dir, f"image_{i}_boxes.png"), model)
        
        # Person extraction & color
        person_cloud = extract_points_in_boxes(current_cloud, predictions, res=0.02, x_range=(-5, 5), y_range=(-5, 5))
        change_colors(person_cloud, COLORS["GREEN"])
        
        # Build static map (gray) and remove person points
        current_cloud_static = remove_points_by_proximity(current_cloud, person_cloud, radius=0.05)
        current_cloud_downsampled = current_cloud_static.voxel_down_sample(voxel_size)
        current_cloud_downsampled, _ = current_cloud_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        change_colors(current_cloud_downsampled, COLORS["GREY"])
        
        # Apply the current transformation to the local clouds (for better ICP initialization)
        current_cloud_downsampled.transform(current_transformation)
        person_cloud.transform(current_transformation)
        
        # Run ICP between the current cloud and the static map 
        icp_result = o3d.pipelines.registration.registration_icp(
            current_cloud_downsampled, static_map, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria = criteria
        )
        
        # Update the current transformation for the next iteration
        current_transformation = np.dot(current_transformation, icp_result.transformation)
        transformations.append(icp_result.transformation)

        # Apply ICP transformation to the current cloud and person cloud
        current_cloud_downsampled.transform(icp_result.transformation)
        person_cloud.transform(icp_result.transformation)
        
        # Trajectory update
        trajectory_cloud.transform(icp_result.transformation)
        change_colors(trajectory_cloud, COLORS["RED"])
        trajectory_global_cloud += trajectory_cloud
        
        # Filter and merge into static map
        filtered_cloud = filter_close_points(static_map, current_cloud_downsampled, CONFIG["radius"])
        static_map += filtered_cloud
        
        # Update the person cloud and the global cloud
        person_global_cloud += person_cloud
        show_cloud += static_map + trajectory_cloud + person_cloud
        
        # print_color_counts(show_cloud, COLORS)
        
        if CONFIG["print_realtime"]:
            display_point_cloud(vis_map, show_cloud, point_size=1.0)
        
        print(f"Cloud {i} aligned. Total number of points in the full map: {len(show_cloud.points)}")
    
    return static_map, trajectory_global_cloud, person_global_cloud

def save_static_map(global_map, timestamp, base_path):
    output_path = os.path.join(base_path, f"{timestamp}_map.ply")
    o3d.io.write_point_cloud(output_path, global_map)
    print(f"Static map saved to {output_path}")

def save_trajectory(global_trajectory, timestamp, base_path):
    output_path = os.path.join(base_path, f"{timestamp}_trajectory.ply")
    o3d.io.write_point_cloud(output_path, global_trajectory)
    print(f"Trajectory map saved to {output_path}")

def save_person_cloud(person_global_cloud, timestamp, base_path):
    output_path = os.path.join(base_path, f"{timestamp}_person.ply")
    o3d.io.write_point_cloud(output_path, person_global_cloud)
    print(f"Person map saved to {output_path}")

def main():
    start_time = time.time()

    # Get CSV file list from selected folder
    folder_path = FILE_NAME[CONFIG["selected_file"]]
    csv_files = get_csv_files(folder_path)
    
    # # Set up visualizer if real-time visualization is enabled
    if CONFIG["print_realtime"]:
        vis_map = setup_visualizer()
    else:
        vis_map = None
    
    # Run alignment
    static_map, global_trajectory, global_person_cloud = icp_alignment(csv_files, folder_path, vis_map)
    
    # Prepare save path and timestamp
    timestamp = datetime.now().strftime("%d%m_%H%M")  # Format: daymonth_hourminute
    base_path = f"C:/Users/Albert/Desktop/lidar-fused-3d-map/src/visualization/saved_maps_and_trajectories/{CONFIG['selected_file']}"
    
    # Save results if enabled
    if CONFIG["save_map"]:
        save_static_map(static_map, timestamp, base_path)
        save_trajectory(global_trajectory, timestamp, base_path)
        save_person_cloud(global_person_cloud, timestamp, base_path)
    
    # Print execution time
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    # Show final visualization if enabled
    if CONFIG["print_realtime"]:
        vis_map.run()   


if __name__ == "__main__":
    clear_terminal()
    main()