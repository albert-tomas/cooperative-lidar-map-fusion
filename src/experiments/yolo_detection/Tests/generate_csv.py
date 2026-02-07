import pandas as pd
import numpy as np

# Generate N random points in a given range
N = 128000
x = np.random.uniform(-5, 5, N)
y = np.random.uniform(-5, 5, N)
z = np.random.uniform(-2, 2, N)

# Fill in dummy values for the other columns
data = {
    "Ch": np.ones(N),
    "AziCorr(deg)": np.ones(N),
    "Dist(m)": np.ones(N),
    "Rfl": np.ones(N),
    "Azi(deg)": np.ones(N),
    "Ele(deg)": np.ones(N),
    "t(us)": np.ones(N),
    "x(m)": x,
    "y(m)": y,
    "z(m)": z,
}

df = pd.DataFrame(data)
df.to_csv("C:/Users/Albert/Desktop/lidar-fused-3d-map/YOLO_file/test_points.csv", index=False)