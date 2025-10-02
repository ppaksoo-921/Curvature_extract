# Curvature_extract
Welcome to the GitHub repository pertaining to our journal: curvature computation of image gather for semi-automated depth image quality control.

Our study refines the Flood Fill algorithm to make it suitable for application to raw data, enabling the extraction of reflection signal curvature from image gathers.

# Files
The repository contains the following files:
* Extract_curvature_Anglegather.py
* Extract_curvature_Anglegather.py
* Various angle gather data files used in our research

# Data availability
Due to data sharing restrictions, the CMP dataset cannot be provided.  
However, to ensure reproducibility, we instead provide the synthetic angle-domain common image gather dataset used for algorithm validation in Data_angle_gather.zip.

# Usage
1. Download and extract **Data_angle_gather.zip**.  
2. Update the dataset path inside `Extract_curvature_Anglegather.py` to point to the extracted files.  
3. Run the script to visualize curvature extraction results:
   ```bash
   python Extract_curvature_Anglegather.py

# Python Libraries
Our research utilizes various Python libraries that need to be installed:
* NumPy
* Matplotlib
* Scipy
* (Additionally uses standard modules: sys, time, itertools)

# Contact Information
* Name: Soo Hwan Park
* Email: ppaksoo@snu.ac.kr
