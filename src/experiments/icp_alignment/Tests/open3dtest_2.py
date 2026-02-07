import sys
import numpy as np
import open3d as o3d
import time
import pandas as pd
import pygetwindow as gw
import os
import copy

def get_csv_files(folder_path):
    # Returns a list of CSV files in the given directory
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    # Check if there are any CSV files 
    if not csv_files:
        print("No CSV files found in the folder.")
    else:
        print(f"Found {len(csv_files)} CSV files.")
    return csv_files

def load_point_clouds(csv_files, folder_path, max_iterations=1):
    # Loads up to max_iterations point clouds from CSV files
    # Each point cloud has the x, y, z coordinates and the file name
    point_clouds = []
    for i, file in enumerate(csv_files):
        if i >= max_iterations:
            break
        full_path = os.path.join(folder_path, file)
        pc = pd.read_csv(full_path, usecols=["x(m)", "y(m)", "z(m)"])
        pc["file"] = file
        point_clouds.append(pc)
        print(i)
    return point_clouds

def create_open3d_point_cloud(df):
    # Converts a DataFrame into an Open3D PointCloud object
    x, y, z = df["x(m)"], df["y(m)"], df["z(m)"]
    points = np.vstack((x, y, z)).T
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    # TODO: maybe return the points as well if they have to be processed after, avoiding recalculating
    return pc

def display_point_cloud(vis, point_cloud, point_size=2.0):
    # Displays the point cloud in the visualizer
    vis.clear_geometries()
    vis.add_geometry(point_cloud)
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    
    time.sleep(1)
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()

# cloud = o3d.geometry.PointCloud()
ply_point_cloud = o3d.data.PLYPointCloud()

pcd = o3d.io.read_point_cloud(ply_point_cloud.path)

folder_path = "D:/lidar-thesis/Capture1911/CSV"
csv_files = get_csv_files(folder_path)

# Load point clouds from CSV files
point_clouds = load_point_clouds(csv_files, folder_path, max_iterations=1)

pc = create_open3d_point_cloud(point_clouds[0])



# Crear el visualizador
visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=800, height=600)
# Get the window and maximize it
window = gw.getWindowsWithTitle('Open3D')[0]
window.maximize()

# Añadir la nube de puntos original al visualizador
#visualizer.add_geometry(pc)

# Actualizar el renderizador
visualizer.update_renderer()

# Mostrar la nube de puntos original durante 2 segundos
visualizer.poll_events()
visualizer.update_renderer()
#time.sleep(2)  # Espera de 2 segundos

# # Ahora actualizamos para mostrar la nube de puntos filtrada
# visualizer.clear_geometries()  # Limpiar la geometría actual

translation_matrix = np.array([
    [1, 0, 0, 10],  # Traslación de 2 unidades en el eje X
    [0, 1, 0, 10],  # Traslación de 3 unidades en el eje Y
    [0, 0, 1, 4],  # Traslación de 4 unidades en el eje Z
    [0, 0, 0, 1]   # Mantener la homogeneidad
])


pc2 = copy.deepcopy(pc)
display_point_cloud(visualizer, pc2) 
time.sleep(2)  # Espera de 2 segundos

pc2.transform(translation_matrix)  # Aplicar la transformación a la nube de puntos


pc.points.extend(pc2.points)
pc3 = pc + pc2
display_point_cloud(visualizer, pc3)  # Añadir la geometría filtrada

# # Actualizar el renderizado
visualizer.update_renderer()

# # Mantener la ventana abierta para la visualización
visualizer.run()