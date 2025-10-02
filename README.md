# Curvature_extract
Welcome to the GitHub repository pertaining to our journal: curvature computation of image gather for semi-automated depth image quality control.

Our study refines the Flood Fill algorithm to make it suitable for application to raw data, enabling the extraction of reflection signal curvature from image gathers.

# Curvature_extract
Welcome to the GitHub repository accompanying our journal: **Curvature computation of image gathers for semi-automated depth image quality control**.

Seismic images and velocity models are essential for subsurface interpretation, but quantitative validation is necessary for more accurate results. Traditional Radon transform methods for curvature estimation often suffer from limitations on real data, such as non-uniqueness, energy smearing, and event overlap.  

In this study, we propose a novel algorithm that adapts the **Flood Fill** technique, commonly used in computer graphics, to isolate individual reflection events from seismic gathers. Curvature is then estimated by computing the quadratic coefficient in a least-squares fit. The method has been validated on both:
- **Normal moveout-corrected CMP gathers** (Yeongil Bay field data)  
- **Angle-domain common image gathers** (synthetic velocity model)  


# Files
The repository contains the following files:
* Extract_curvature_Anglegather.py
* Extract_curvature_Anglegather.py
* Various angle gather data files used in our research

# Python Libraries

The following Python libraries are required to run the code:

### Core libraries
These are essential for numerical computation and visualization:
- **NumPy** (`import numpy as np`) – array operations and mathematical functions  
- **Matplotlib** (`import matplotlib.pyplot as plt`) – plotting and visualization  

### Utility libraries
These are used for general-purpose support:
- **os** – file system operations (directory creation, path manipulation)  
- **sys** – system configuration (e.g., recursion limit setting)  
- **time** – execution time measurement  
- **itertools** (`from itertools import product`) – combinatorial operations  

### Optional
These libraries provide additional functionality but are not strictly required:
- **PIL** (`from PIL import Image`) – image processing  
- **matplotlib.font_manager** (`import matplotlib.font_manager as fm`) – font management for custom plots  

# Data availability
Due to data sharing restrictions, the CMP dataset cannot be provided.

However, to ensure reproducibility, we instead provide the synthetic angle-domain common image gather dataset used for algorithm validation in **Data_angle_gather.zip**.

# Usage
1. Download and extract **Data_angle_gather.zip**.  
2. Update the dataset path inside `Extract_curvature_Anglegather.py` to point to the extracted files.  
3. Run the script to visualize curvature extraction results:
   ```bash
   python Extract_curvature_Anglegather.py

# Contact Information
* Name: Soo Hwan Park
* Email: ppaksoo@snu.ac.kr
