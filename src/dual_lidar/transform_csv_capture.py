import os
import numpy as np
import pandas as pd
import open3d as o3d

def get_csv_files(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the folder.")
    else:
        print(f"Found {len(csv_files)} CSV files.")
    return csv_files

def apply_initial_transformation(points, rotation_deg):
    theta = np.radians(rotation_deg)
    transform = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])
    
    # Añadir columna de 1s para aplicar transformación homogénea
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))  # Nx4
    transformed_points_h = points_h @ transform.T  # Transformación
    return transformed_points_h[:, :3]  # Solo x, y, z

def transform_csv_file(input_path, output_path, rotation_deg):
    df = pd.read_csv(input_path)

    if not {"x(m)", "y(m)", "z(m)"}.issubset(df.columns):
        print(f"Skipping file (missing required columns): {input_path}")
        return

    # Extraer y transformar coordenadas
    xyz = df[["x(m)", "y(m)", "z(m)"]].values
    xyz_transformed = apply_initial_transformation(xyz, rotation_deg)

    # Reemplazar columnas en el DataFrame original
    df[["x(m)", "y(m)", "z(m)"]] = xyz_transformed

    # Guardar
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    original_csv_path = "D:/LiDAR-captures/Strada3/CSV"
    output_csv_path = "D:/LiDAR-captures/Strada3/CSV_transformed"
    os.makedirs(output_csv_path, exist_ok=True)

    csv_files = get_csv_files(original_csv_path)

    for idx, file in enumerate(csv_files):
        print(f"Processing file {idx}: {file}")
        input_file = os.path.join(original_csv_path, file)
        output_file = os.path.join(output_csv_path, file)
        transform_csv_file(input_file, output_file, rotation_deg=180)