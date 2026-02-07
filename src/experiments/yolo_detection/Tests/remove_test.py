import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from roboflow import Roboflow
import open3d as o3d
import pygetwindow as gw



def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')    

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
    # TODO: we could invert the order, first filter and then digitize, to make it faster

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

def yolo_add_boxes():
    # Image path
    image_file = "C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/output_bev_image_083.png"
    image = cv2.imread(image_file)

    # Initialize Roboflow
    rf = Roboflow(api_key="m3mVIcAJXOFxaob0eVAt")

    # Load the trained model
    model = rf.workspace("sicariata").project("person_lidar").version(1).model

    # Make the prediction
    predictions = model.predict(image, confidence=40, overlap=30).json()

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
    cv2.imwrite("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/output_bev_image_083_with_boxes.png", image)

    # Print success message
    print("Image saved with predictions.")
    
    return predictions

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

def remove_points_in_boxes(csv_path, predictions, output_csv_path, res=0.02, x_range=(-5, 5), y_range=(-5, 5)):
    # Load the original CSV
    df = pd.read_csv(csv_path)

    # Original BEV image dimensions
    bev_width = int((x_range[1] - x_range[0]) / res)
    bev_height = int((y_range[1] - y_range[0]) / res)

    # Assumed YOLO image size (change if you used a different size!)
    yolo_width = 600
    yolo_height = 600

    scale_x = bev_width / yolo_width
    scale_y = bev_height / yolo_height

    for pred in predictions['predictions']:
        # Box coordinates in pixels
        x1_px = int((pred['x'] - pred['width'] / 2) * scale_x)
        x2_px = int((pred['x'] + pred['width'] / 2) * scale_x)
        y1_px = int((pred['y'] - pred['height'] / 2) * scale_y)
        y2_px = int((pred['y'] + pred['height'] / 2) * scale_y)

        # Convert pixels to real-world coordinates
        # Note: the Y-axis in the image grows downward, and in BEV it does too, so no need to invert it
        x_min = x_range[0] + x1_px * res
        x_max = x_range[0] + x2_px * res
        y_min = y_range[0] + y1_px * res
        y_max = y_range[0] + y2_px * res

        inside = df[((df['x(m)'] >= x_min) & (df['x(m)'] <= x_max) &
            (df['y(m)'] >= y_min) & (df['y(m)'] <= y_max))]
        print(f"Points to remove in box {x_min:.2f}-{x_max:.2f}, {y_min:.2f}-{y_max:.2f}: {len(inside)}")

        # Filter out points inside the box (keep only the ones outside)
        df = df[~((df['x(m)'] >= x_min) & (df['x(m)'] <= x_max) &
                (df['y(m)'] >= y_min) & (df['y(m)'] <= y_max))]

    # Save the modified CSV
    df.to_csv(output_csv_path, index=False)
    print(f"New CSV saved without points inside the boxes: {output_csv_path}")

if __name__ == "__main__":
    clear_terminal()
    # Convert initial CSV to PNG
    # Show CSV
    #show_csv("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/frame_083.csv")
    create_bev_image('C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/frame_083.csv', 'C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/output_bev_image_083.png')
    # Detect objects from PNG with YOLO
    predictions = yolo_add_boxes()
    #print(predictions['predictions'])
    # Remove points from CSV
    remove_points_in_boxes('C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/test_points.csv', predictions, 'C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/test_points_removed.csv')
    #show_csv("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/test_points_removed.csv")
    #remove_points_in_boxes('C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/frame_083.csv', predictions, 'C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/frame_083_no_person.csv')
    show_csv("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/frame_083_no_person.csv")
    