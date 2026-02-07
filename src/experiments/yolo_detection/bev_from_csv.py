import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This file is used to create a bird's eye view (BEV) image from a CSV file containing 3D point cloud data

def create_bev_image(csv_file, output_image_path, res=0.02, x_range=(-3, 5), y_range=(-6, 5), z_range=(-2, 2)):
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


# Example usage
# create_bev_image('D:/lidar-thesis/CATTURE_UFFICIO/catture_6_11_24/points_info_20200520_001818_799.csv', 'D:/lidar-thesis/CSV_examples/output_bev_image_try.png')
create_bev_image('D:/lidar-thesis/CSV_examples/my_test/frame_083.csv', 'D:/lidar-thesis/CSV_examples/my_test/output_bev_image_083.png')