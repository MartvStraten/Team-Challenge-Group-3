# Library imports
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
from image_utils import*


def make_input_3D_ROI(image, segment_data, label, euler_angles, sagittal_angle=0, coronal_angle=0, axial_angle=0):
    """ Generates a 3D Region of Interest (ROI) from an input image and segmentation data, 
    while applying specified rotations and updating ground truth angles.
    ---
    Parameters:
        image (numpy.ndarray): The 3D input image volume.
        segment_data (numpy.ndarray): The segmentation data corresponding to the input image.
        label (int): The label of the target vertebra in the segmentation data.
        euler_angles (dict): A dictionary containing the ground truth Euler angles for each label.
        saggital_angle (float, optional): The rotation angle (in degrees) around the sagittal axis. Default is 0.
        coronal_angle (float, optional): The rotation angle (in degrees) around the coronal axis. Default is 0.
        axial_angle (float, optional): The rotation angle (in degrees) around the axial axis. Default is 0.
    ---
    Output:
            - ROI_3D_image (numpy.ndarray): The 3D ROI image extracted after applying the random rotations.
            - ref_3D_image (numpy.ndarray): The reference 3D ROI image based on updated ground truth rotations.
            - rotated_image (numpy.ndarray): The rotated version of the input image.
            - rotated_mask (numpy.ndarray): The rotated segmentation.
    """

    vertebra_mask = (segment_data == label).astype(int)
    # Update the ground truth angles based on rotation of the 3D input volume
    updated_euler_angle = euler_angles[label].copy()
    if sagittal_angle != 0:
        updated_euler_angle[0] = updated_euler_angle[0] - sagittal_angle
    # Uncomment if you want to apply coronal angles
    #if coronal_angle != 0:
    #    updated_euler_angle[1] = updated_euler_angle[1] - coronal_angle
    if axial_angle != 0:
        updated_euler_angle[2] = updated_euler_angle[2] - axial_angle
    
    # Rotate the input 3D ROI
    rotated_image = rotate_3D(image, sagittal_angle=sagittal_angle,axial_angle=axial_angle)
    rotated_mask = rotate_3D(vertebra_mask, sagittal_angle=sagittal_angle, axial_angle=axial_angle)
    # Obtain individual segmented vertebra bbox image
    rot_bbox_coord, rot_bbox_mask = compute_bbox(rotated_mask)
    ROI_3D_image = rotated_image[
        rot_bbox_coord["z_min"]-5:rot_bbox_coord["z_max"]+5,
        rot_bbox_coord["y_min"]:rot_bbox_coord["y_max"],
        rot_bbox_coord["x_min"]:rot_bbox_coord["x_max"],
    ]

    # Obtain the reference ground truth rotations
    ref_rotated_image = rotate_3D(rotated_image, sagittal_angle=updated_euler_angle[0], axial_angle=updated_euler_angle[2])
    ref_rotated_mask = rotate_3D(rotated_mask, sagittal_angle=updated_euler_angle[0], axial_angle=updated_euler_angle[2])
    ref_rot_bbox_coord, ref_rot_bbox_mask = compute_bbox(ref_rotated_mask)
    ref_3D_image = ref_rotated_image[
        ref_rot_bbox_coord["z_min"]-5:ref_rot_bbox_coord["z_max"]+5,
        ref_rot_bbox_coord["y_min"]:ref_rot_bbox_coord["y_max"],
        ref_rot_bbox_coord["x_min"]:ref_rot_bbox_coord["x_max"],
    ]
    
    return ROI_3D_image, ref_3D_image, rotated_image, rotated_mask

def generate_2d_slices(image, angles):
    """ Generates a 2D slice from a 3D image by applying rotations and compressing along an axis.
    ---
    Parameters:
        image (numpy.ndarray): The 3D input image volume.
        angles (list or tuple of float): A list or tuple containing Euler angles [sagittal, coronal, axial] 
                                         for rotation in degrees.
    ---
    Output:
        image_2d (numpy.ndarray): A 2D compressed representation of the rotated 3D image.
    """
    # Rotate the image and compress along the coronal axis (2)
    rotated_image = rotate_3D(image, sagittal_angle = angles[0], axial_angle =angles[2]) 
    image_2d = compress_bonemri(rotated_image, axis=2).astype(np.float32) 
    
    return image_2d

def template_matching_score(image_2d, template):
    """Calculates the matching score between a 2D image and a template using the normalized cross-correlation.
    ---
    Parameters:
        image_2d (numpy.ndarray): The input 2D image in which the template will be searched.
        template (numpy.ndarray): The template image to be matched within the 2D image.
    ---
    Output:
        max_val (float): The maximum matching score (ranging from -1 to 1), where 1 indicates a perfect match.
    """
    # Calculate the template matching score using normalized cross-correlation
    result = cv2.matchTemplate(image_2d, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    return max_val

def loss_function(angles, image, template):
    """Computes the loss for template matching optimization by comparing the 2D projection 
    of the image with a template using a matching score. Invert this score to minimize the loss.
    ---
    Parameters:
        angles (list or tuple of float): A list or tuple containing Euler angles [sagittal, coronal, axial] 
                                         used to generate a 2D slice from the 3D image.
        image (numpy.ndarray): The 3D input image to be rotated and sliced.
        template (numpy.ndarray): The template image to match against the generated 2D slice.
    ---
    Output:
        score (float): The loss value (negative of the template matching score) to be minimized during optimization.
    ---
    Note:
        The function skips the angles for optimization if the image size is smaller than the template size.
    """
    # Generate 2D slice from the 3D image using the specified angles
    image_2d = generate_2d_slices(image, angles)
    # If the image is smaller than the template, skip this iteration
    if image_2d.shape[0] < template.shape[0] or image_2d.shape[1] < template.shape[1]:
        score = inf
        return score  # Return a large value to indicate an invalid score (skipping this iteration)
    # Compute the template matching score
    score = template_matching_score(image_2d, template)
    # Return the negative score to minimize the loss
    return -score

def optimize_rotation(ROI_3D_image, template, bounds = [(-45, 45), (0, 0), (-45, 45)]):
    """Optimizes the rotation of a 3D Region of Interest (ROI) to maximize the matching score
    between the 2D projection of the ROI and a given template using differential evolution.
    ---
    Parameters:
        ROI_3D_image (numpy.ndarray): The 3D image of the Region of Interest (ROI) to be rotated.
        template (numpy.ndarray): The 2D template image to match against the projected 2D slice of the ROI.
        bounds (list of tuples, optional): A list of bounds for the rotation angles.
    ---
    Output:                         
        numpy.ndarray: The optimal rotation angles that maximize the template matching score.
    """
    # Run the optimization
    result = opt.differential_evolution(
        loss_function, 
        bounds, 
        args=(ROI_3D_image, template),
        strategy='best1bin',  
        maxiter=100, 
        popsize=15,  
        tol=1e-6,  
        mutation=(0.5, 1), 
        recombination=0.7  
    )
    return result.x

def get_euler_angles(segment_data):
    """Computes the Euler angles and bounding box coordinates for each vertebra in the segmentation data using PCA.
    ---
    Parameters:
        segment_data (numpy.ndarray): The segmentation data corresponding to the input image.
    ---
    Output:
        bounding_box_coords (dict): A dictionary mapping each vertebra label to its corresponding 
                                    bounding box coordinates (min and max values for each axis).
        bounding_box_masks (dict): A dictionary mapping each vertebra label to its binary mask.
        euler_angles (dict): A dictionary mapping each vertebra label to its computed Euler angles 
    """
    # Create the bounding boxes around the segmented verebrae
    vertebrae_labels = np.unique(segment_data)[1:] # Obtain unique labels for the vertebrae (skipping zero)

    # Find bounding cube coordinates for each vertebrae
    bounding_box_coords = {}
    bounding_box_masks = {}
    for label in vertebrae_labels:
        # Get bounding box
        vertebra_mask = (segment_data == label)
        bbox_coord, bbox_mask = compute_bbox(vertebra_mask)

        # Store bounding box data
        bounding_box_coords[int(label)] = bbox_coord
        bounding_box_masks[int(label)] = bbox_mask

    # Compute the principal components and their rotation angles for each bounding box
    centered_points = {}
    principal_components = {}
    euler_angles = {}

    for label in vertebrae_labels:
        # Create a binary mask for the specific vertebra
        vertebra_mask = (segment_data == label)

        # Compute the oriented bounding box (OBB)
        points, pc, angles = compute_pca(vertebra_mask, int(label))

        # Store results
        centered_points[int(label)] = points
        principal_components[int(label)] = pc
        euler_angles[int(label)] = angles

    # print(f"Computed bounding box for {len(vertebrae_labels)} vertebra")

    return bounding_box_coords, bounding_box_masks, euler_angles

def generate_template(image, segment_data, label, euler_angles, plot=False):
    """Generates a 2D template image based on the rotation via the Euler angles.
    ---
    Parameters:
        image (numpy.ndarray): The 3D input image volume from which the template will be generated.
        segment_data (numpy.ndarray): The segmentation data corresponding to the input image.
        label (int): The label of the vertebra for which the template is generated.
        euler_angles (dict): A dictionary containing the Euler angles for each vertebra label.
        plot (bool, optional): If True, plots the generated 2D image and the cropped masked image for visualization. 
    ---
    Output:
        numpy.ndarray: The 2D template image of the rotated vertebra region, with central masking applied.
    """
    # Rotate the 3D image to the ground truth position to obtain perfect face
    rotated_image = rotate_3D(image, sagittal_angle=euler_angles[label][0], axial_angle=euler_angles[label][2])
    # Rotate the segmentation to the ground truth position
    vertebra_mask = (segment_data == label).astype(int)
    rotated_mask = rotate_3D(vertebra_mask, sagittal_angle=euler_angles[label][0], axial_angle=euler_angles[label][2])
    rot_bbox_coord, rot_bbox_mask = compute_bbox(rotated_mask)
    ROI_3D_image = rotated_image[
        rot_bbox_coord["z_min"]-5:rot_bbox_coord["z_max"]+5,
        rot_bbox_coord["y_min"]:rot_bbox_coord["y_max"],
        rot_bbox_coord["x_min"]:rot_bbox_coord["x_max"],
    ]

    # Obtain the ground truth image by compressing the 3D rotated image
    image_2D = compress_bonemri(ROI_3D_image, axis=2).astype(np.float32)
    # Create a mask to generate a template image
    mask = np.zeros_like(image_2D, dtype=int)
    mask[10:image_2D.shape[0]-10, 10:image_2D.shape[1]-10] = 1
    masked = image_2D*mask
    # Crop the mask image in order to be smaller then the comparison image for template matching
    cropped_mask = masked[10:image_2D.shape[0]-10, 10:image_2D.shape[1]-10] 

    # If plot is true, plot the ground truth image with the generated template image
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].imshow(np.rot90(image_2D,3), cmap='gray')
        ax[1].imshow(np.rot90(cropped_mask,3), cmap='gray')
    
    template_image = np.array(cropped_mask, dtype=np.float32)

    return template_image