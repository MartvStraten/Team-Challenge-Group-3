# Library imports
import numpy as np
import pydicom
import os

def load_3d_dicom(patient_path):
    # Create dicom file paths 
    filenames = os.listdir(patient_path)

    # Load all DICOM data 
    dicom_files = [pydicom.dcmread(os.path.join(patient_path, f)) for f in filenames]

    # Sort slices by image position
    dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[0]))

    # Convert to 3D numpy array
    image = np.stack([
        f.pixel_array*f.RescaleSlope + f.RescaleIntercept 
        for f in dicom_files
    ], axis=0)

    return image

def apply_window(image, window_level, window_width):
    # Apply window and normalize to 0-255
    lower = window_level - window_width/2
    upper = window_level + window_width/2
    image = np.clip(image, lower, upper)
    image = (image - lower)/(upper - lower)*255

    return image
