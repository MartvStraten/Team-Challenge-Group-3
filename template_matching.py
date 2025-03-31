# Library imports
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
from image_utils import*


def make_input_3D_ROI(image, segment_data, label, euler_angles, saggital_angle=0, coronal_angle=0, axial_angle=0):
    
    vertebra_mask = (segment_data == label).astype(int)
    # Update the ground truth angles based on rotation of the 3D input volume
    updated_euler_angle = euler_angles[label].copy()
    if saggital_angle != 0:
        updated_euler_angle[0] = updated_euler_angle[0] - saggital_angle
    #if coronal_angle != 0:
    #    updated_euler_angle[1] = updated_euler_angle[1] - coronal_angle
    if axial_angle != 0:
        updated_euler_angle[2] = updated_euler_angle[2] - axial_angle
    print("The updated euler angles after initial rotation are: ", updated_euler_angle)
    # Rotate the input 3D ROI
    rotated_image = rotate_3D(image, sagittal_angle=saggital_angle, axial_angle=axial_angle)
    rotated_mask = rotate_3D(vertebra_mask, sagittal_angle=saggital_angle, axial_angle=axial_angle)
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


# Function to generate 2D slices
def generate_2d_slices(image, angles):
    rotated_image = rotate_3D(image, sagittal_angle = angles[0], axial_angle =angles[2])
    image_2d = compress_bonemri(rotated_image, axis=2).astype(np.float32)
    return image_2d

# Function to calculate template matching score
def template_matching_score(image_2d, template):
    result = cv2.matchTemplate(image_2d, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val

# Loss function for optimization
def loss_function(angles, image, template):
    image_2d = generate_2d_slices(image, angles)
    score = template_matching_score(image_2d, template)
    return -score  # Minimize the negative score to maximize the template matching score

def optimize_rotation(ROI_3D_image, template, bounds = [(-45, 45), (0, 0), (-45, 45)], strategy='best1bin', maxiter=100, popsize=15, tol=1e-6, mutation=(0.5, 1), recombination=0.7):
    
    # Run the optimization
    result = opt.differential_evolution(
        loss_function, 
        bounds, 
        args=(ROI_3D_image, template),
        strategy='best1bin',  # Standaard strategie, kan aangepast worden
        maxiter=100,  # Aantal iteraties
        popsize=15,  # Populatiegrootte
        tol=1e-6,  # Convergentietolerantie
        mutation=(0.5, 1),  # Mutatie parameters
        recombination=0.7  # Kruisingsparameter
    )
    return result.x

def get_euler_angles(segment_data):
    
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
    # Store results for each vertebra
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

    print(f"Computed bounding box for {len(vertebrae_labels)} vertebra")

    return bounding_box_coords, bounding_box_masks, euler_angles

def generate_template(image, segment_data, label, euler_angles, plot=False):
    
    rotated_image = rotate_3D(image, sagittal_angle=euler_angles[label][0], axial_angle=euler_angles[label][2])
    vertebra_mask = (segment_data == label).astype(int)
    rotated_mask = rotate_3D(vertebra_mask, sagittal_angle=euler_angles[label][0], axial_angle=euler_angles[label][2])
    rot_bbox_coord, rot_bbox_mask = compute_bbox(rotated_mask)
    ROI_3D_image = rotated_image[
        rot_bbox_coord["z_min"]-5:rot_bbox_coord["z_max"]+5,
        rot_bbox_coord["y_min"]:rot_bbox_coord["y_max"],
        rot_bbox_coord["x_min"]:rot_bbox_coord["x_max"],
    ]


    image_2D = compress_bonemri(ROI_3D_image, axis=2).astype(np.float32)
    mask = np.zeros_like(image_2D, dtype=int)
    mask[10:image_2D.shape[0]-10, 10:image_2D.shape[1]-10] = 1
    masked = image_2D*mask
    cropped_mask = masked[10:image_2D.shape[0]-10, 10:image_2D.shape[1]-10] 

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].imshow(np.rot90(image_2D,3), cmap='gray')
        ax[1].imshow(np.rot90(cropped_mask,3), cmap='gray')
    
    template_image = np.array(cropped_mask, dtype=np.float32)
    return template_image