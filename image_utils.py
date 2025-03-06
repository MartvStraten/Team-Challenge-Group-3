# Library imports
import numpy as np
import pydicom
import cv2
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

def apply_window(image, window_level=450, window_width=1500):
    # Apply window and normalize to 0-255
    lower = window_level - window_width/2
    upper = window_level + window_width/2
    image = np.clip(image, lower, upper)
    image = (image - lower)/(upper - lower)*255

    return image

def compress_bonemri(image, axis, threshold_min=100, threshold_max=255):    
    # Apply thresholding (set values outside range to 0)
    binary_image = np.where((image >= threshold_min) & (image <= threshold_max), image, 0)

    # Compress the 3D image to 2D using max projection
    compressed_image = np.sum(binary_image, axis=axis)

    return compressed_image

def pad_to_cube(image, axis, target_depth=None):
    """Pads a 3D image to a cube shape (target_size, target_size, target_size)."""
    if axis not in [0, 1, 2]:
        raise ValueError("Axis must be 0, 1, or 2")
    
    if target_depth is None:
        target_depth = max(image.shape)
        
    current_depth = image.shape[axis]
    
    if current_depth >= target_depth:
        return image  # No padding needed if already large enough
    
    # Compute padding for depth (only pad the first axis)
    pad_before = (target_depth - current_depth) // 2
    pad_after = target_depth - current_depth - pad_before
    
    min_intensity = np.min(image)  # Use min intensity instead of 0 to avoid artifacts

    if target_depth is None:
        target_depth = max(image.shape)
    
    if axis == 0:
        # Apply padding only along depth (first axis)
        padded_image = np.pad(image, [(pad_before, pad_after), (0, 0), (0, 0)], 
                                    mode='constant', constant_values=min_intensity)   
    elif axis == 1:
        # Apply padding only along width (second axis)
        padded_image = np.pad(image, [(0, 0), (pad_before, pad_after), (0, 0)], 
                                    mode='constant', constant_values=min_intensity)
    else:
        # Apply padding only along height (third axis)
        padded_image = np.pad(image, [(0, 0), (0, 0), (pad_before, pad_after)], 
                                    mode='constant', constant_values=min_intensity) 
    
    return padded_image

def rotate_image(image: np.ndarray, axis: int, angle: float):
    """
    Rotate a 3D image along a given axis using OpenCV for faster processing.

    Parameters
    ----------
    image : np.ndarray
        The 3D image to rotate.
    axis : int
        The axis along which to rotate the image (0 = sagittal, 1 = coronal, 2 = axial).
    angle : float
        The angle (in degrees) by which to rotate the image. (Can be positive or negative)

    Returns
    -------
    np.ndarray
        The rotated 3D image.
    """
    # Pad image to cube before rotation
    image = pad_to_cube(image, axis=0, target_depth=max(image.shape))
    
    # Get image shape
    d, h, w = image.shape

    # Select rotation plane
    if axis == 0:  # Sagittal: Rotate along Y-Z
        slices = [image[i, :, :] for i in range(d)]
    elif axis == 1:  # Coronal: Rotate along X-Z
        slices = [image[:, i, :] for i in range(h)]
    elif axis == 2:  # Axial: Rotate along X-Y
        slices = [image[:, :, i] for i in range(w)]
    else:
        raise ValueError("Invalid axis. Use 0, 1, or 2.")
    
    # Get rotation matrix for 2D rotation
    center = (slices[0].shape[1] // 2, slices[0].shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation to each slice
    rotated_slices = [
        cv2.warpAffine(slice_, rotation_matrix, (slice_.shape[1], slice_.shape[0]), flags=cv2.INTER_LINEAR)
        for slice_ in slices
    ]

    # Stack back to 3D
    if axis == 0:
        rotated_image = np.stack(rotated_slices, axis=0)
    elif axis == 1:
        rotated_image = np.stack(rotated_slices, axis=1)
    elif axis == 2:
        rotated_image = np.stack(rotated_slices, axis=2)

    return rotated_image