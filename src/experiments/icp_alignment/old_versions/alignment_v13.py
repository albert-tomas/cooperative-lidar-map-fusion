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



# v13 - Added visualization of the 3 maps (the global map, the trajectory and the person cloud)


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
    # "save_trajectory": False,
    "radius": 0.2, # Radius for filtering close points
}

COLORS = {
    "RED": [1, 0, 0],
    "GREEN": [0, 1, 0],
    "BLUE": [0, 0, 1],
    "GREY": [0.5, 0.5, 0.5],
}

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')
    #print(CONFIG["selected_file"])

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
    # Image path
    #image_file = "C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/output_bev_image_083.png"
    image = cv2.imread(image_file)

    # # Initialize Roboflow
    # rf = Roboflow(api_key="m3mVIcAJXOFxaob0eVAt")

    # # Load the trained model
    # model = rf.workspace("sicariata").project("person_lidar").version(1).model

    # Make the prediction
    predictions = model.predict(image, confidence=10, overlap=30).json()

    # Draw the boxes on the image
    for prediction in predictions['predictions']:
        x1 = int(prediction['x'] - prediction['width'] / 2)
        y1 = int(prediction['y'] - prediction['height'] / 2)
        x2 = int(prediction['x'] + prediction['width'] / 2)
        y2 = int(prediction['y'] + prediction['height'] / 2)

        # Draw the rectangle on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # BGR, 2 is the line thickness

        # Draw the class label and confidence
        label = f"{prediction['class']} ({prediction['confidence']*100:.2f}%)"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image with the predictions
    cv2.imwrite(output_image_path, image)

    # Print success message
    # print("Image saved with predictions.")
    
    return predictions

def highliht_points_in_boxes(point_cloud, predictions,  highlight_color, res=0.02, x_range=(-5, 5), y_range=(-5, 5)): 
    
    # Convert the point cloud to numpy for easier indexing
    # points = np.asarray(point_cloud.points)
    # colors = np.asarray(point_cloud.colors)

    # # Image scale factors
    # bev_width = int((x_range[1] - x_range[0]) / res)
    # bev_height = int((y_range[1] - y_range[0]) / res)
    # yolo_width, yolo_height = 600, 600
    # scale_x = bev_width / yolo_width
    # scale_y = bev_height / yolo_height

    # for pred in predictions["predictions"]:
    #     # Convert bounding box from YOLO image to real-world coordinates
    #     x1_px = (pred["x"] - pred["width"] / 2) * scale_x
    #     x2_px = (pred["x"] + pred["width"] / 2) * scale_x
    #     y1_px = (pred["y"] - pred["height"] / 2) * scale_y
    #     y2_px = (pred["y"] + pred["height"] / 2) * scale_y

    #     y1_idx = bev_height - y2_px
    #     y2_idx = bev_height - y1_px

    #     x_min = x_range[0] + x1_px * res
    #     x_max = x_range[0] + x2_px * res
    #     y_min = y_range[0] + y1_idx * res
    #     y_max = y_range[0] + y2_idx * res

    #     # Find and color the points within the box
    #     in_box = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
    #             (points[:, 1] >= y_min) & (points[:, 1] <= y_max))
    #     colors[in_box] = highlight_color  # Paint those points

    # point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

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

def count_points_by_color(pcd, color_map, eps=1e-3):
    cols = np.asarray(pcd.colors)
    counts = {}
    for name, rgb in color_map.items():
        mask = np.all(np.abs(cols - rgb) < eps, axis=1)
        counts[name] = int(mask.sum())
    return counts

def icp_alignment(csv_files, folder_path, vis_map):
    threshold = 0.5  # Distance threshold for ICP matching. Correspondance distance
    max_iterations = 50 # Maximum number of iterations for ICP
    max_number_of_clouds = CONFIG["max_number_of_clouds"]
    voxel_size = CONFIG["voxel_size"]
    
    # Step 1: Create the initial global map (reference cloud)
    input_cloud = load_csv_as_open3d_point_cloud(csv_files[0], folder_path)
    
    # Make a BEV image of the first cloud
    bev_images_path = "D:/LiDAR-captures/" + CONFIG["selected_file"] + "/bev_images/image_0.png"
    input_path = os.path.join(folder_path, csv_files[0])
    create_bev_image(input_path, bev_images_path, res=0.02, x_range=(-5, 5), y_range=(-5, 5), z_range=(-2, 2))
    
    model = initialize_roboflow()
    # Make a BEV image with boxes based on YOLO predictions
    bev_images_path_output = "D:/LiDAR-captures/" + CONFIG["selected_file"] + "/bev_images/image_0_boxes.png"
    predictions = yolo_add_boxes(bev_images_path, bev_images_path_output, model)
    
    #TODO
    person_cloud = extract_points_in_boxes(input_cloud, predictions, res=0.02, x_range=(-5, 5), y_range=(-5, 5))
    change_colors(person_cloud, COLORS["GREEN"])
    person_global_cloud = o3d.geometry.PointCloud()
    person_global_cloud += person_cloud
    
    global_map = input_cloud.voxel_down_sample(voxel_size)
    change_colors(global_map, COLORS["GREY"])
    
    trajectory_transformed_cloud = generate_single_point()
    change_colors(trajectory_transformed_cloud, COLORS["BLUE"])
    trajectory_global_cloud = o3d.geometry.PointCloud()
    trajectory_global_cloud += trajectory_transformed_cloud

    show_cloud = o3d.geometry.PointCloud()
    show_cloud += global_map + trajectory_transformed_cloud + person_cloud

    # Display the initial point cloud
    if CONFIG["print_realtime"]:
        color_stats = count_points_by_color(show_cloud,
            {"grey":  COLORS["GREY"],
            "green": COLORS["GREEN"],
            "blue": COLORS["BLUE"],
            "red": COLORS["RED"]})

        print(f"Frame: "
            f"\t\t{color_stats['grey']} grey,"
            f"\t\t{color_stats['green']} green,"
            f"\t\t{color_stats['blue']} blue,"
            f"\t\t{color_stats['red']} red")
        
        display_point_cloud(vis_map, show_cloud, point_size=1.0)
    
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iterations)
    
    # Initialize an empty list to store transformations
    transformations = []    
    # Initial transformation (identity matrix)
    current_transformation = np.eye(4)

    
    # Iterate over the next point clouds
    for i in range(1, min(len(csv_files), max_number_of_clouds)):
        current_cloud = load_csv_as_open3d_point_cloud(csv_files[i], folder_path)
        current_cloud_downsampled = current_cloud.voxel_down_sample(voxel_size)
        
        #Generate BEV image for current cloud
        bev_images_path = "D:/LiDAR-captures/" + CONFIG["selected_file"] + "/bev_images/image_" + str(i) + ".png"
        input_path = os.path.join(folder_path, csv_files[i])
        create_bev_image(input_path, bev_images_path, res=0.02, x_range=(-5, 5), y_range=(-5, 5), z_range=(-2, 2))
        
        # Make a BEV image with boxes based on YOLO predictions
        bev_images_path_output = "D:/LiDAR-captures/" + CONFIG["selected_file"] + "/bev_images/image_" + str(i) + "_boxes.png"
        predictions = yolo_add_boxes(bev_images_path, bev_images_path_output, model)
        # current_cloud_downsampled = highliht_points_in_boxes(current_cloud_downsampled, predictions, COLORS["GREEN"], res=0.02, x_range=(-5, 5), y_range=(-5, 5))
        #TODO
        person_cloud = extract_points_in_boxes(input_cloud, predictions, res=0.02, x_range=(-5, 5), y_range=(-5, 5))
        change_colors(person_cloud, COLORS["GREEN"])
        
        #global_map += current_cloud
        #display_point_cloud(vis, global_map, point_size=2.0)
        change_colors(current_cloud_downsampled, COLORS["GREY"])
        
        # Apply the current transformation to the local cloud
        current_cloud_downsampled.transform(current_transformation)
        person_cloud.transform(current_transformation)
        
        # Run ICP between the current cloud and the global map 
        icp_result = o3d.pipelines.registration.registration_icp(
            current_cloud_downsampled, global_map, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria = criteria
        )
        
        current_transformation = np.dot(current_transformation, icp_result.transformation)

        # Transform the current cloud and add it to the global map
        current_cloud_downsampled.transform(icp_result.transformation)
        person_cloud.transform(icp_result.transformation)
        # Store the transformation
        transformations.append(icp_result.transformation)
        
        # Transform the trajectory point cloud
        trajectory_transformed_cloud.transform(icp_result.transformation)
        change_colors(trajectory_transformed_cloud, COLORS["RED"])
        trajectory_global_cloud += trajectory_transformed_cloud
        #change_colors(trajectory_global_cloud, COLORS["RED"])
        
        # Filter points that are too close to existing points in the global map
        filtered_cloud = filter_close_points(global_map, current_cloud_downsampled, CONFIG["radius"])
        # change_colors(filtered_cloud, COLORS["GREY"])

        # Add the transformed cloud to the global map
        global_map += filtered_cloud
        person_global_cloud += person_cloud
        
        show_cloud += global_map + trajectory_transformed_cloud + person_cloud
        if CONFIG["print_realtime"]:
            color_stats = count_points_by_color(show_cloud,
            {"grey":  COLORS["GREY"],
            "green": COLORS["GREEN"],
            "blue": COLORS["BLUE"],
            "red": COLORS["RED"]})

            print(f"Frame: "
                f"\t\t{color_stats['grey']} grey,"
                f"\t\t{color_stats['green']} green,"
                f"\t\t{color_stats['blue']} blue,"
                f"\t\t{color_stats['red']} red")
            display_point_cloud(vis_map, show_cloud, point_size=1.0)
        
        print(f"Cloud {i} aligned. Total number of points in the full map: {len(show_cloud.points)}")
    
    return global_map, trajectory_global_cloud, person_global_cloud

def save_map(global_map, timestamp):
    # Save the global map to a file
    folder_path_map = f"C:/Users/Albert/Desktop/lidar-fused-3d-map/src/visualization/saved_maps_and_trajectories/{CONFIG['selected_file']}"
    
    output_path = os.path.join(folder_path_map, f"map_{timestamp}.ply")
    o3d.io.write_point_cloud(output_path, global_map)
    print(f"Global map saved to {output_path}")

def save_trajectory(global_trajectory, timestamp):
    folder_path_trajectory = f"C:/Users/Albert/Desktop/lidar-fused-3d-map/src/visualization/saved_maps_and_trajectories/{CONFIG['selected_file']}"
    
    output_path = os.path.join(folder_path_trajectory, f"trajectory_{timestamp}.ply")
    o3d.io.write_point_cloud(output_path, global_trajectory)
    print(f"Trajectory saved to {output_path}")

def save_person_cloud(person_global_cloud, timestamp):
    folder_path_trajectory = f"C:/Users/Albert/Desktop/lidar-fused-3d-map/src/visualization/saved_maps_and_trajectories/{CONFIG['selected_file']}"
    
    output_path = os.path.join(folder_path_trajectory, f"trajectory_{timestamp}.ply")
    o3d.io.write_point_cloud(output_path, person_global_cloud)
    print(f"Trajectory saved to {output_path}")

def main():
    start_time = time.time()
    
    # Variables
    
    
    # Folder path with all the CSV files
    #folder_path = "C:/Users/Albert/Desktop/lidar-fused-3d-map/CSVfiles"
    folder_path = FILE_NAME[CONFIG["selected_file"]]
    # folder_path = "D:/lidar-thesis/PCAP_CSV_files/Capture1911/CSV"
    csv_files = get_csv_files(folder_path)
    
    # Initialize and run the visualizer
    if CONFIG["print_realtime"]:
        vis_map = setup_visualizer()
    else:
        vis_map = None
    global_map, global_trajectory, global_person_cloud = icp_alignment(csv_files, folder_path, vis_map)
    
    # Get the current date and time
    timestamp = datetime.now().strftime("%d%m_%H%M")  # Format: daymonth_hourminute
    
    if CONFIG["save_map"]:
        save_map(global_map, timestamp)
        save_trajectory(global_trajectory, timestamp)
        save_person_cloud(global_person_cloud, timestamp)
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    if CONFIG["print_realtime"]:
        vis_map.run()   


if __name__ == "__main__":
    clear_terminal()
    main()