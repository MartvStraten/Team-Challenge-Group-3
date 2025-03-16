# Library imports
import SimpleITK as sitk
import numpy as np

from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

def load_3d_dicom(dicom_path):
    """Loads a 3D dicom image.
    ---
    Parameters:
        dicom_path (string): directory to dicom folder
    ---
    Output:
        ct_image (np.ndarray): 3D CT image
    """
    # Initiate dicom reader object
    reader = sitk.ImageSeriesReader()
    # Load dicom series using reader
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_series)
    # Execute reader to obtain CT image
    ct_image = reader.Execute()

    return sitk.GetArrayFromImage(ct_image)

def apply_window(ct_image, window_level=450, window_width=1500):
    """"Applies a windowing and normalisation to image intensities.
    ---
    Parameters:
        ct_image (np.ndarray): 3D CT image
        window_level (int): center of intensity window
        window_width (int): width of intensity window
    ---
    Output:
        ct_image_windowed (np.ndarray): windowed 3D CT image
    """
    # Calculate bounds of intensity window
    lower = window_level - window_width/2
    upper = window_level + window_width/2
    # Apply pixel intensity window
    ct_image_windowed = np.clip(ct_image, lower, upper)
    # Apply pixel intensity normalisation
    ct_image_windowed = (ct_image_windowed - lower)/(upper - lower)*255

    return ct_image_windowed

def compress_bonemri(ct_image, axis=2, threshold_min=0):  
    """Applies threshold to 3D CT and compresses image to 2D.
    ---
    Parameters:
        ct_image (np.ndarray): 3D CT image
        axis (int): 0,1,2 = sagittal, axial, coronal
        threshold_min (int): minimum intensity value
    ---
    Output:
        compressed_image (np.ndarray): 2D compressed image
    """  
    # Apply thresholding (set values outside range to 0)
    binary_image = np.where(ct_image > threshold_min, ct_image, 0)
    # Compress the 3D image to 2D using max projection
    compressed_image = np.sum(binary_image, axis=axis)

    return compressed_image

def get_radiograph(ct_image, axis=0, threshold_min=0): 
    """Generate 2D radiograph from 3D CT image.
    ---
    Parameters:
        ct_image (np.ndarray): 3D CT image
        axis (int): 0,1,2 = coronal, axial, sagittal
        threshold_min (int): minimum intensity value
    ---
    Output:
        radiograph (np.ndarray): 2D radiograph image
    """
    # Apply thresholds
    bone_mask = ct_image > threshold_min
    ct_image[~bone_mask] = np.min(ct_image)
    # Generate radiograph
    radiograph_sitk = sitk.MaximumProjection(sitk.GetImageFromArray(ct_image), axis)

    return np.squeeze(sitk.GetArrayFromImage(radiograph_sitk))

def rotate_3D(image, sagittal_angle=0.0, coronal_angle=0.0, axial_angle=0.0):
    """Rotates a 3D image in arbitrary axis.
    ---
    Parameters:
        image (np.ndarray): 3D image
        sagittal_angle (float): rotation angle in sagittal plane
        coronal_angle (float): rotation angle in coronal plane
        axial_angle (float): rotation angle in axial plane
    ---
    Output:
        resampled_image (np.ndarray): rotated 3D image
    """
    # Transform image to SimpleITK object
    image = sitk.GetImageFromArray(image)
    
    # Obtain information for rotation
    size = image.GetSize()
    center = [s / 2.0 for s in size]

    # Initialize transform object
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)
    transform.SetRotation(np.deg2rad(coronal_angle), np.deg2rad(axial_angle), np.deg2rad(sagittal_angle))

    # Perform rotation and resample the image
    resampled_image = sitk.Resample(
        image, 
        image.GetSize(), 
        transform, 
        sitk.sitkLinear, 
        image.GetOrigin(), 
        image.GetSpacing(), 
        image.GetDirection(), 
        0, 
        image.GetPixelID()
    )

    return sitk.GetArrayFromImage(resampled_image)

def compute_bbox(binary_volume):
    """Compute the bounding box around a binary volume.
    ---
    Parameters:
        binary_volume (np.ndarray): a 3D binary volume (vertebra)
    ---
    Output:
        bbox_coord (dict): coordinates corners bounding box
        bbox_mask (np.ndarray): binary mask containing bounding box
    """
    # Get indices for individual vertebra
    indices_vertebra = np.where(binary_volume)

    # Get minimum and maximum coordinates in all three dimensions
    x_min, x_max = np.min(indices_vertebra[2]), np.max(indices_vertebra[2])
    y_min, y_max = np.min(indices_vertebra[1]), np.max(indices_vertebra[1])
    z_min, z_max = np.min(indices_vertebra[0]), np.max(indices_vertebra[0])

    # Store bounding cube coordinates
    bbox_coord = {
        "x_min": int(x_min), "x_max": int(x_max),
        "y_min": int(y_min), "y_max": int(y_max),
        "z_min": int(z_min), "z_max": int(z_max)
    }

    # Create and store bounding cube
    bbox_mask = np.zeros_like(binary_volume, dtype=int)
    bbox_mask[z_min:z_max, y_min:y_max, x_min:x_max] = 1

    return bbox_coord, bbox_mask

def compute_pca(binary_volume, label):
    """Compute the principle components and their rotation angles.
    ---
    Parameters:
        binary_volume (np.ndarray): a 3D binary volume (vertebra)
        label (int): vertebra label
    ---
    Returns:
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix.
        euler_angles (tuple): (yaw, pitch, roll) in degrees.
    """
    # Get voxel coordinates
    coords = np.argwhere(binary_volume)
    centered_coords = coords - coords.mean(axis=0)
    # Cut off spinous process for lumbar vertebrae
    if label >= 12:
        coords = coords[coords[:, 1] < np.percentile(coords[:, 1], 95)]

    # Compute PCA
    pca = PCA(n_components=3)
    pca.fit(coords)
    components = pca.components_

    # Ensure Y-axis is correctly oriented (fix flipped axis issue)
    if np.linalg.det(components) < 0:
        components[1, :] *= -1  # Flip second principal component if needed
    
    # Convert to Euler angles (XYZ order)
    euler_angles = R.from_matrix(components).as_euler('xyz', degrees=True)
    
    # Ensure angles are between -45 and 45 degrees
    euler_angles = np.array([(((a + 45)  % 90) - 45) for a in euler_angles])

    return centered_coords, components, euler_angles