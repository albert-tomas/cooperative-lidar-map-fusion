import sys
import numpy as np
import open3d as o3d
import time
import pygetwindow as gw


# cloud = o3d.geometry.PointCloud()
ply_point_cloud = o3d.data.PLYPointCloud()

pcd = o3d.io.read_point_cloud(ply_point_cloud.path)

# # Convertir la nube de puntos a un array de numpy para facilitar el filtrado
# points = np.asarray(pcd.points)
# # Calcular el umbral Z (puedes usar la mediana o el valor medio de Z)
# z = points[:, 2]  # Extraer la coordenada Z de cada punto
# z_threshold = np.median(z)  # Usamos la mediana como umbral

# # Filtrar los puntos, eliminando aquellos cuya coordenada Z esté por encima del umbral
# filtered_points = points[z < z_threshold]

# # Crear una nueva nube de puntos con los puntos filtrados
# filtered_pcd = o3d.geometry.PointCloud()
# filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)



# Crear el visualizador
visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=800, height=600)
# Get the window and maximize it
window = gw.getWindowsWithTitle('Open3D')[0]
window.maximize()

# Añadir la nube de puntos original al visualizador
visualizer.add_geometry(pcd)

# Actualizar el renderizador
visualizer.update_renderer()

# Mostrar la nube de puntos original durante 2 segundos
visualizer.poll_events()
visualizer.update_renderer()
time.sleep(2)  # Espera de 2 segundos

# Ahora actualizamos para mostrar la nube de puntos filtrada
visualizer.clear_geometries()  # Limpiar la geometría actual

translation_matrix = np.array([
    [1, 0, 0, 2],  # Traslación de 2 unidades en el eje X
    [0, 1, 0, 3],  # Traslación de 3 unidades en el eje Y
    [0, 0, 1, 4],  # Traslación de 4 unidades en el eje Z
    [0, 0, 0, 1]   # Mantener la homogeneidad
])

pcd2 = pcd
pcd2.transform(translation_matrix)  # Aplicar la transformación a la nube de puntos


pcd.points.extend(pcd2.points)

visualizer.add_geometry(pcd)  # Añadir la geometría filtrada

# Actualizar el renderizado
visualizer.update_renderer()

# Mantener la ventana abierta para la visualización
visualizer.run()