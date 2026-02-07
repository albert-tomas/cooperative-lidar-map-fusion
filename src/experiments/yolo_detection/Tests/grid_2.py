import os
import pandas as pd
import numpy as np
import open3d as o3d

def create_xy_grid(x_range=(-5, 5), y_range=(-5, 5), step=1.0, z_level=0.0):
    lines = []
    points = []
    # vertical lines
    for x in np.arange(x_range[0], x_range[1]+step, step):
        points.append([x, y_range[0], z_level])
        points.append([x, y_range[1], z_level])
        lines.append([len(points)-2, len(points)-1])
    # horizontal lines
    for y in np.arange(y_range[0], y_range[1]+step, step):
        points.append([x_range[0], y, z_level])
        points.append([x_range[1], y, z_level])
        lines.append([len(points)-2, len(points)-1])
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines  = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector([[0.5,0.5,0.5] for _ in lines])
    return grid

def show_csv(file_path):
    # 1. Cargo CSV y creo PointCloud
    df = pd.read_csv(file_path, usecols=["x(m)","y(m)","z(m)"])
    pts = np.vstack((df["x(m)"], df["y(m)"], df["z(m)"])).T
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    # 2. Creo rejilla y ejes
    grid = create_xy_grid(x_range=(-10,10), y_range=(-10,10), step=1.0)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])

    # 3. Dibujo todo junto
    o3d.visualization.draw_geometries(
        [pc, grid, axes],
        window_name="Point Cloud with Grid and Axes",
        width=800, height=600
    )

if __name__ == "__main__":
    show_csv("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/solo_test/solo_person.csv")