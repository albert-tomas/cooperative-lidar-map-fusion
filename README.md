# Cooperative LiDAR Sensing and Map Fusion

This repository contains the code developed for my Bachelor’s Degree Thesis (TFG) in
Telecommunications Engineering, titled *Cooperative LiDAR Sensing in Vehicular Scenarios*.

The project explores a cooperative perception approach based on LiDAR data sharing,
with the objective of generating a consistent 3D map from two independent mobile robots.
The implementation follows an experimental, research-oriented methodology.
This work was developed during an Erasmus exchange at Politecnico di Milano, in collaboration with the DEIB laboratory.

## Project Overview

Single-LiDAR perception systems suffer from limited field of view, occlusions, and blind
spots, especially in urban or dynamic environments. This work investigates a cooperative
LiDAR-based pipeline where information from two mobile robots is combined to improve
environmental awareness and mapping accuracy.


The implemented pipeline includes the following stages:

1. Offline preprocessing of LiDAR point clouds stored in CSV format
2. Detection and removal of dynamic objects (pedestrians) using YOLO applied to BEV images
3. Frame-to-map alignment using ICP-based registration
4. Trajectory estimation through accumulated transformations
5. Dual-robot map fusion using relative pose estimation and ICP refinement
6. Visualization of maps and robot trajectories

The system is implemented in Python and focuses on accurate alignment and map consistency,
rather than real-time performance.


## Example Result

Below is an example of the final fused static map obtained by aligning and merging
two LiDAR captures using object-based initialization and ICP refinement.

![Original and fused LiDAR maps](images/fused_map.png)

## Repository Structure

```
src/
├── core/
│   ├── dual_lidar_fusion.py        # Dual-robot map fusion pipeline
│   ├── icp_alignment.py            # ICP-based point cloud alignment
│   ├── remove_yolo.py              # Dynamic object removal using YOLO detections
│   └── transform_csv_capture.py    # LiDAR CSV preprocessing
│
├── visualization/
│   └── map_and_trajectory_viewer.py
│
└── experiments/
    ├── dual_lidar/
    ├── icp_alignment/
    ├── yolo_detection/
    └── YOLO_file/
```

- **`core/`** contains the scripts corresponding to the final pipeline used to obtain the
  results presented in the thesis.
- **`visualization/`** provides tools to inspect point clouds, maps, and estimated trajectories.
- **`experiments/`** contains intermediate versions, tests, datasets, and experimental artifacts
  generated during the iterative development process.

## Notes

This repository reflects an academic research workflow.
Multiple script versions and experimental components are intentionally preserved to
document the evolution of the proposed methods.

Some large data files and trained models used during experimentation are included only
partially or for reference purposes.

## Related Documents

- Bachelor’s Thesis (PDF):  
  [Cooperative LiDAR Sensing in Vehicular Scenarios]()

- Thesis presentation slides (PDF):  
  [Project presentation]()

## Author

Albert Tomàs Ruiz  
Bachelor’s Degree in Telecommunications Engineering  
Universitat Politècnica de Catalunya (UPC)