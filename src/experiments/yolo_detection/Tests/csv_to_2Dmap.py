import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#Read csv file
data = pd.read_csv('D:\PointClouds\CSV\points_info_20200520_000807_399.csv')
print(f"Columns: {data.columns}")
x, y, z = data['x(m)'], data['y(m)'], data['z(m)']

#Create a 2D grid (bird's eye view)
grid_size = 256  # Dimensione della griglia
depth_map = np.zeros((grid_size, grid_size))

#Normalize the coordinates to adjust them to the grid
x_norm = ((x - x.min()) / (x.max() - x.min()) * (grid_size - 1)).astype(int)
y_norm = ((y - y.min()) / (y.max() - y.min()) * (grid_size - 1)).astype(int)

#Fill matrix with z values
for i in range(len(x)):
    depth_map[y_norm[i], x_norm[i]] = z[i]
    
#Visualize and save as a color map
plt.imshow(depth_map, cmap='gray')
barra = plt.colorbar()
barra.set_label("Altezza_relativa(m)")

#Save the image as a PNG using the CSV file's name
#output_file = os.path.join(output_path, f"{os.path.basename(file_name).replace('.csv', '')}_depth_image.png")
#plt.savefig(output_file)
plt.imsave('D:\PointClouds\CSV\points_info_20200520_000807_399.png', depth_map, cmap='gray')
plt.show()

#Close the figure to avoid overlaps between iterations
plt.clf()

print("Process finished, image saved.")