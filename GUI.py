import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import random
import cv2
import os

import tkinter as tk
import customtkinter as ctk

from tkinter import filedialog
from PIL import Image, ImageTk

from image_utils import *
from template_matching import *

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        # Configure window -------------------------------------------------------------------------------------------------------
        self.title("Team Challenge Group 3")
        self.geometry(f"{1100}x{580}")

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # LEFT SIDEBAR -----------------------------------------------------------------------------------------------------------
        self.sidebar_frame = ctk.CTkFrame(self, width=180, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=9, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        # Create "Team Challenge" logo
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, 
            text="Team Challenge", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Create "Select CT image" button left sidebar
        self.dicom_path = None
        self.sidebar_button_1 = ctk.CTkButton(self.sidebar_frame, 
            text="Select DICOM folder", 
            command=self.select_patient_ct
        )
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.patient_ct_text = ctk.CTkLabel(self.sidebar_frame, 
            text="No folder selected",
            anchor="w"
        )
        self.patient_ct_text.grid(row=2, column=0, padx=20, pady=(0, 10))

        # Create "Select segmentation" button left sidebar
        self.segmentation_path = None
        self.sidebar_button_2 = ctk.CTkButton(self.sidebar_frame, 
            text="Select segmentation", 
            command=self.select_segmentation
        )
        self.sidebar_button_2.grid(row=3, column=0, padx=20, pady=10)
        self.segmentation_text = ctk.CTkLabel(self.sidebar_frame, 
            text="No file selected",
            anchor="w"
        )
        self.segmentation_text.grid(row=4, column=0, padx=20, pady=(0, 10))

        # Create "Load data" button left sidebar
        self.sidebar_button_3 = ctk.CTkButton(self.sidebar_frame, 
            text="Load data", 
            command=self.load_data,
            state="disabled"
        )
        self.sidebar_button_3.grid(row=5, column=0, padx=20, pady=10)

        # Create "Appearance mode:" text left sidebar
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, 
            text="Appearance Mode:", 
            anchor="w"
        )
        self.appearance_mode_label.grid(row=7, column=0, padx=20, pady=(10, 0))

        # Create appearance mode optionmenu button left sidebar
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, 
            values=["Dark", "Light", "System"],
            command=self.change_appearance_mode_event
        )
        self.appearance_mode_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 10))

        # BOTTOM ENTRY WIDGET AND BUTTON -----------------------------------------------------------------------------------------
        self.entry = ctk.CTkEntry(self, 
            placeholder_text="Search"
        )
        self.entry.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.main_button_1 = ctk.CTkButton(self,
            text="Enter", 
            fg_color="transparent", 
            border_width=2, 
            text_color=("gray10", "#DCE4EE")
        )
        self.main_button_1.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # DICOM VIEWER ------------------------------------------------------------------------------------------------------------
        self.image_tabview = ctk.CTkTabview(self, corner_radius=0)
        self.image_tabview.grid(row=0, column=1, padx=(20,0), pady=(20, 0), sticky="nsew") 

        self.image_tabview.add("DICOM viewer")
        self.dicom_viewer_frame = self.image_tabview.tab("DICOM viewer")
        self.dicom_viewer_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.dicom_viewer_frame.grid_rowconfigure((0, 1, 2, 3), weight=1)

        # Sagittal DICOM image label
        self.sagittal_dicom_image_label = tk.Label(self.dicom_viewer_frame,
            bg="gray10"
        )
        self.sagittal_dicom_image_label.grid(row=0, column=0, padx=10, pady=10)
        # Coronal DICOM image label
        self.coronal_dicom_image_label = tk.Label(self.dicom_viewer_frame, 
            bg="gray10"
        )
        self.coronal_dicom_image_label.grid(row=0, column=1, padx=10, pady=10)
        # Axial DICOM image label
        self.axial_dicom_image_label = tk.Label(self.dicom_viewer_frame, 
            bg="gray10"
        )
        self.axial_dicom_image_label.grid(row=0, column=2, padx=10, pady=10)

        # Start slice indices in the middle
        self.sagittal_slice_idx = 50
        self.coronal_slice_idx = 336
        self.axial_slice_idx = 336

        # Sagittal slider widget
        self.sagittal_slider = ctk.CTkSlider(self.dicom_viewer_frame, 
            from_=0, 
            to=100, 
            command=self.update_slice_sagittal
        )
        self.sagittal_slider_text = ctk.CTkLabel(self.dicom_viewer_frame, 
            text=f"Sagittal slice {self.sagittal_slice_idx}"
        )
        # Coronal slider widget
        self.coronal_slider = ctk.CTkSlider(self.dicom_viewer_frame, 
            from_=0, 
            to=671, 
            command=self.update_slice_coronal
        )
        self.coronal_slider_text = ctk.CTkLabel(self.dicom_viewer_frame, 
            text=f"Coronal slice {self.coronal_slice_idx}"
        )
        # Axial slider widget
        self.axial_slider = ctk.CTkSlider(self.dicom_viewer_frame, 
            from_=0, 
            to=671, 
            command=self.update_slice_axial
        )
        self.axial_slider_text = ctk.CTkLabel(self.dicom_viewer_frame, 
            text=f"Axial slice {self.axial_slice_idx}"
        )

        # RADIOGRAPH VIEWER ------------------------------------------------------------------------------------------------------
        self.image_tabview.add("Radiograph viewer")
        self.radiograph_viewer_frame = self.image_tabview.tab("Radiograph viewer")
        self.radiograph_viewer_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.radiograph_viewer_frame.grid_rowconfigure((0, 1, 2, 3), weight=1)

        # Sagittal radiograph image label
        self.sagittal_radiograph_image_label = tk.Label(self.radiograph_viewer_frame,
            bg="gray10"
        )
        self.sagittal_radiograph_image_label.grid(row=0, column=0, padx=10, pady=10)
        # Coronal radiograph image label
        self.coronal_radiograph_image_label = tk.Label(self.radiograph_viewer_frame, 
            bg="gray10"
        )
        self.coronal_radiograph_image_label.grid(row=0, column=1, padx=10, pady=10)
        # Axial radiograph image label
        self.axial_radiograph_image_label = tk.Label(self.radiograph_viewer_frame, 
            bg="gray10"
        )
        self.axial_radiograph_image_label.grid(row=0, column=2, padx=10, pady=10)

        # Start angle at 0 degrees
        self.sagittal_radiograph_angle = 0.0
        self.coronal_radiograph_angle = 0.0
        self.axial_radiograph_angle = 0.0

        # Sagittal rotation slider widget
        self.sagittal_radiograph_slider = ctk.CTkSlider(self.radiograph_viewer_frame, 
            from_=-45, 
            to=45, 
            number_of_steps=900,
            command=self.update_radiograph_angle_sagittal
        )
        self.sagittal_radiograph_slider_text = ctk.CTkLabel(self.radiograph_viewer_frame, 
            text=f"Sagittal angle: {self.sagittal_radiograph_angle:.1f} degrees"
        )
        # Coronal slider widget
        self.coronal_radiograph_slider = ctk.CTkSlider(self.radiograph_viewer_frame, 
            from_=-15, 
            to=15, 
            number_of_steps=300,
            command=self.update_radiograph_angle_coronal
        )
        self.coronal_radiograph_slider_text = ctk.CTkLabel(self.radiograph_viewer_frame, 
            text=f"Coronal angle: {self.coronal_radiograph_angle:.1f} degrees"
        )
        # Axial slider widget
        self.axial_radiograph_slider = ctk.CTkSlider(self.radiograph_viewer_frame, 
            from_=-15, 
            to=15, 
            number_of_steps=300,
            command=self.update_radiograph_angle_axial
        )
        self.axial_radiograph_slider_text = ctk.CTkLabel(self.radiograph_viewer_frame, 
            text=f"Axial angle: {self.axial_radiograph_angle:.1f} degrees"
        )

        # VERTEBRA VIEWER -----------------------------------------------------------------------------------------------
        self.image_tabview.add("Vertebra viewer")
        self.vertebra_viewer_frame = self.image_tabview.tab("Vertebra viewer")
        self.vertebra_viewer_frame.grid_columnconfigure((0, 1), weight=1)
        self.vertebra_viewer_frame.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)

        # Radiograph-based bbox text label
        self.radiograph_bbox_text_label = ctk.CTkLabel(self.vertebra_viewer_frame, 
            text="Current angles",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.radiograph_bbox_text_label.grid(row=0, column=0, padx=20, pady=(0, 10))
        # Radiograph-based bbox label
        self.radiograph_bbox_label = tk.Label(self.vertebra_viewer_frame,
            bg="gray10"
        )
        self.radiograph_bbox_label.grid(row=1, column=0, padx=10, pady=10)
        # Text labels to indicate current angles
        self.vertebra_sag_angle_text_label = ctk.CTkLabel(self.vertebra_viewer_frame, 
            text=f"Current sagittal angle: {self.sagittal_radiograph_angle} degrees"
        )
        self.vertebra_sag_angle_text_label.grid(row=2, column=0, padx=20, pady=10)
        self.vertebra_cor_angle_text_label = ctk.CTkLabel(self.vertebra_viewer_frame, 
            text=f"Current coronal angle: {self.coronal_radiograph_angle} degrees"
        )
        self.vertebra_cor_angle_text_label.grid(row=3, column=0, padx=20, pady=10)
        self.vertebra_ax_angle_text_label = ctk.CTkLabel(self.vertebra_viewer_frame, 
            text=f"Current axial angle: {self.axial_radiograph_angle} degrees"
        )
        self.vertebra_ax_angle_text_label.grid(row=4, column=0, padx=20, pady=10)

        # Optimized bbox text label
        self.optimized_bbox_text_label = ctk.CTkLabel(self.vertebra_viewer_frame, 
            text="Optimized angles",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.optimized_bbox_text_label.grid(row=0, column=1, padx=20, pady=(0, 10))
        # Optimized bbox label
        self.optimized_bbox_label = tk.Label(self.vertebra_viewer_frame, 
            bg="gray10"
        )
        self.optimized_bbox_label.grid(row=1, column=1, columnspan=3, padx=10, pady=10)
        # Text labels to indicate optimized angles
        self.opt_sag_angle_text_label = ctk.CTkLabel(self.vertebra_viewer_frame, 
            text="Optimized sagittal angle: 0 degrees"
        )
        self.opt_sag_angle_text_label.grid(row=2, column=1, padx=20, pady=10)
        self.opt_cor_angle_text_label = ctk.CTkLabel(self.vertebra_viewer_frame, 
            text=f"Optimized coronal angle: 0 degrees"
        )
        self.opt_cor_angle_text_label.grid(row=3, column=1, padx=20, pady=10)
        self.opt_ax_angle_text_label = ctk.CTkLabel(self.vertebra_viewer_frame, 
            text=f"Optimized axial angle: 0 degrees"
        )
        self.opt_ax_angle_text_label.grid(row=4, column=1, padx=20, pady=10)

        # RIGHT SIDE BAR ---------------------------------------------------------------------------------------------------------
        self.right_tabview = ctk.CTkTabview(self, width=300)
        self.right_tabview.grid(row=0, column=3, padx=(20, 10), pady=(20, 0), sticky="nsew")

        self.right_tabview.add("Features")
        self.features_frame = self.right_tabview.tab("Features")
        self.features_frame.grid_columnconfigure(0, weight=1) 
        self.features_frame.grid_rowconfigure((0, 1), weight=0) 

        # Window level and width variables
        self.window_level = 450
        self.window_width = 1500
        # DICOM Image text label
        self.dicom_image_text = ctk.CTkLabel(self.features_frame,
            text="----- DICOM Image -----",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.dicom_image_text.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")
        # Window level text label
        self.window_level_text = ctk.CTkLabel(self.features_frame,
            text=f"Window level: {self.window_level}"
        )
        self.window_level_text.grid(row=1, column=0, padx=20, pady=0, sticky="nsew")
        # Window level slider widget
        self.window_level_slider = ctk.CTkSlider(self.features_frame, 
            from_=0, 
            to=900, 
            number_of_steps=90,
            command=self.update_window_level
        )
        self.window_level_slider.grid(row=2, column=0, padx=10, pady=(0,5), sticky="ew")
        # Window width text label
        self.window_width_text = ctk.CTkLabel(self.features_frame,
            text=f"Window width: {self.window_width}"
        )
        self.window_width_text.grid(row=3, column=0, padx=20, pady=0, sticky="nsew")
        # Window width slider widget
        self.window_width_slider = ctk.CTkSlider(self.features_frame, 
            from_=0, 
            to=3000, 
            number_of_steps=300,
            command=self.update_window_width
        )
        self.window_width_slider.grid(row=4, column=0, padx=10, pady=(0,10), sticky="ew")
        # Apply window button
        self.update_window_button = ctk.CTkButton(self.features_frame,
            text="Apply window", 
            command=self.update_window,
            state="disabled"
        )
        self.update_window_button.grid(row=5, column=0, padx=20, pady=(10,5))
        # Reset window button
        self.reset_window_button = ctk.CTkButton(self.features_frame,
            text="Reset window", 
            command=self.reset_window,
            state="disabled"
        )
        self.reset_window_button.grid(row=6, column=0, padx=20, pady=(5,10))

        # Text label radiographs
        self.radiograph_text = ctk.CTkLabel(self.features_frame,
            text="----- Radiograph -----",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.radiograph_text.grid(row=7, column=0, padx=20, pady=10, sticky="nsew")
        # Option menu for segmented radiograph
        self.segment_radiograph = ["Full radiograph", "Spine-only radiograph"]
        self.optionmenu_radiograph = ctk.CTkOptionMenu(self.features_frame,
            values=self.segment_radiograph,
            command=self.radiograph_select
        )
        self.optionmenu_radiograph.grid(row=8, column=0, padx=10, pady=(5,5), sticky="ew")
        # Option menu for style radiograph
        self.radiograph_styles = ["Fluoroscopy style", "Maximum projection style"]
        self.optionmenu_radiograph_style = ctk.CTkOptionMenu(self.features_frame,
            values=self.radiograph_styles,
            command=self.radiograph_select_style
        )
        self.optionmenu_radiograph_style.grid(row=9, column=0, padx=10, pady=(5,10), sticky="ew")
        # Generate radiograph buttom
        self.generate_radiograph_button = ctk.CTkButton(self.features_frame,
            text="Generate radiograph", 
            command=self.generate_radiograph,
            state="disabled"
        )
        self.generate_radiograph_button.grid(row=10, column=0, padx=20, pady=10)

        # Text label angle calculation
        self.angle_calculation_text = ctk.CTkLabel(self.features_frame,
            text="----- Angle calculation -----",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.angle_calculation_text.grid(row=11, column=0, padx=20, pady=10, sticky="nsew")
        # Target vertebra text
        self.target_vertebra = None
        self.target_vertebra_text = ctk.CTkLabel(self.features_frame,
            text="Select target vertebra"
        )
        self.target_vertebra_text.grid(row=12, column=0, padx=20, pady=0, sticky="nsew")
        # Option menu for target vertebra
        self.vertebra = ["None", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
        self.optionmenu_vertebra = ctk.CTkOptionMenu(self.features_frame,
            values=self.vertebra,
            command=self.vertebra_select
        )
        self.optionmenu_vertebra.grid(row=13, column=0, padx=10, pady=(0, 10), sticky="ew")
        # Calculate reference true AP angles
        self.euler_angles = None
        self.reference_trueAP_button = ctk.CTkButton(self.features_frame,
            text="Calculate reference angles", 
            command=self.calculate_reference_trueAP_angles,
            state="disabled"
        )
        self.reference_trueAP_button.grid(row=14, column=0, padx=20, pady=(10, 5))
        # Get new random radiograph 
        self.random_radiograph_button = ctk.CTkButton(self.features_frame,
            text="Generate random angles", 
            command=self.generate_random_angles,
            state="disabled"
        )
        self.random_radiograph_button.grid(row=15, column=0, padx=20, pady=(5, 5))
        # Start angle optimization
        self.optimize_angles_button = ctk.CTkButton(self.features_frame,
            text="Optimize angles", 
            command=self.optimize_angles,
            state="disabled"
        )
        self.optimize_angles_button.grid(row=16, column=0, padx=20, pady=(5, 10))


    def change_appearance_mode_event(self, new_appearance_mode):
        """Changes GUI appearance."""
        ctk.set_appearance_mode(new_appearance_mode)

    def select_patient_ct(self):
        """Open directory dialog and store the selected folder path."""
        folder_path = filedialog.askdirectory(title="Select DICOM folder")
        if folder_path:
            self.dicom_path = folder_path
            self.patient_ct_text.configure(text=os.path.basename(folder_path))
        if (self.dicom_path and self.segmentation_path):
            self.sidebar_button_3.configure(state="normal")

    def select_segmentation(self):
        """Open file dialog and store the selected file path."""
        file_path = filedialog.askopenfilename(title="Select segmentation file")
        if file_path:
            patient_id = "_".join(os.path.basename(file_path).split("_")[:-1])
            if not file_path.endswith(".nii"):
                self.segmentation_text.configure(text="Please select a .nii file.")
            elif patient_id != os.path.basename(self.dicom_path):
                self.segmentation_text.configure(text="Wrong patient.")
            else:
                self.segmentation_path = file_path
                self.segmentation_text.configure(text=patient_id)
                if (self.dicom_path and self.segmentation_path):
                    self.sidebar_button_3.configure(state="normal")

    def load_data(self):
        """Load patient data and update image frame."""
        # Load dicom data and apply windowing
        self.dicom_image = load_3d_dicom(self.dicom_path)
        self.window_dicom_image = apply_window(self.dicom_image, 
            window_level=self.window_level, 
            window_width=self.window_width
        )

        # Load segmentation data
        segment = sitk.ReadImage(self.segmentation_path)
        self.segment_data = sitk.GetArrayFromImage(segment)
        self.vertebra_mask = np.ones(self.segment_data.shape)
        segment_all = self.segment_data > 0
        self.segment_dicom_image = self.window_dicom_image*segment_all

        # Calculate reference bbox and euler angles for all vertebrae
        self.bbox_coords, self.bbox_masks, self.euler_angles = get_euler_angles(self.segment_data)

        # Set slider range based on the number of slices
        self.sagittal_slider.configure(to=self.dicom_image.shape[0] - 1)
        self.coronal_slider.configure(to=self.dicom_image.shape[1] - 1)
        self.axial_slider.configure(to=self.dicom_image.shape[2] - 1)

        # Prepare & place image in GUI
        self.update_slice_sagittal(self.sagittal_slice_idx)
        self.update_slice_axial(self.axial_slice_idx)
        self.update_slice_coronal(self.coronal_slice_idx)

        # Pack sliders
        self.sagittal_slider.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.coronal_slider.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.axial_slider.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.sagittal_slider_text.grid(row=1, column=2)
        self.coronal_slider_text.grid(row=2, column=2)
        self.axial_slider_text.grid(row=3, column=2)

        self.sagittal_radiograph_slider.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.coronal_radiograph_slider.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.axial_radiograph_slider.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.sagittal_radiograph_slider_text.grid(row=1, column=2)
        self.coronal_radiograph_slider_text.grid(row=2, column=2)
        self.axial_radiograph_slider_text.grid(row=3, column=2)

        # Activate buttons
        self.generate_radiograph_button.configure(state="normal")
        self.update_window_button.configure(state="normal")
        self.reset_window_button.configure(state="normal")

        # Other stuff
        self.dicom_for_radiograph = self.window_dicom_image # Initialize full radiograph
        self.style_radiograph = self.radiograph_styles[0] # Initialize compressed radiograph
        self.generate_radiograph() # Generate initial radiograph

    def update_slice_sagittal(self, value):
        """Update the displayed slice based on the slider value."""
        self.sagittal_slice_idx = int(float(value))
        self.upload_image_to_label(
            self.sagittal_dicom_image_label, 
            self.window_dicom_image[self.sagittal_slice_idx,:,:],
            self.vertebra_mask[self.sagittal_slice_idx,:,:]
        )
        self.sagittal_slider_text.configure(text=f"Sagittal slice {self.sagittal_slice_idx}")

    def update_slice_axial(self, value):
        """Update the displayed slice based on the slider value."""
        self.axial_slice_idx = int(float(value))
        self.upload_image_to_label(
            self.axial_dicom_image_label, 
            np.rot90(self.window_dicom_image[:,self.axial_slice_idx,:], 3),
            np.rot90(self.vertebra_mask[:,self.axial_slice_idx,:], 3)
        )
        self.axial_slider_text.configure(text=f"Axial slice {self.axial_slice_idx}")

    def update_slice_coronal(self, value):
        """Update the displayed slice based on the slider value."""
        self.coronal_slice_idx = int(float(value))
        self.upload_image_to_label(
            self.coronal_dicom_image_label, 
            np.rot90(self.window_dicom_image[:,:,self.coronal_slice_idx], 3),
            np.rot90(self.vertebra_mask[:,:,self.coronal_slice_idx], 3)
        )
        self.coronal_slider_text.configure(text=f"Coronal slice {self.coronal_slice_idx}")

    def update_window_level(self, value):
        """Update output of the window level slider."""
        self.window_level = int(float(value))
        self.window_level_text.configure(text=f"Window level: {self.window_level}")

    def update_window_width(self, value):
        """Update output of the window width slider."""
        self.window_width = int(float(value))
        self.window_width_text.configure(text=f"Window width: {self.window_width}")

    def update_window(self):
        """Update the window settings and DICOM viewer."""
        # Update window of dicom image
        self.window_dicom_image = apply_window(self.dicom_image, 
            window_level=self.window_level, 
            window_width=self.window_width
        )
        # Update DICOM viewer
        self.upload_image_to_label(
            self.sagittal_dicom_image_label, 
            self.window_dicom_image[self.sagittal_slice_idx,:,:],
            self.vertebra_mask[self.sagittal_slice_idx,:,:]
        )
        self.upload_image_to_label(
            self.axial_dicom_image_label, 
            np.rot90(self.window_dicom_image[:,self.axial_slice_idx,:], 3),
            np.rot90(self.vertebra_mask[:,self.axial_slice_idx,:], 3)
        )
        self.upload_image_to_label(
            self.coronal_dicom_image_label, 
            np.rot90(self.window_dicom_image[:,:,self.coronal_slice_idx], 3),
            np.rot90(self.vertebra_mask[:,:,self.coronal_slice_idx], 3)
        )

    def reset_window(self):
        """Reset the window settings and DICOM viewer."""
        # Reset to default values
        self.window_level = 450
        self.window_width = 1500

        # Update the slider positions
        self.window_level_slider.set(self.window_level)
        self.window_width_slider.set(self.window_width)

        # Update the displayed text (if needed)
        self.window_level_text.configure(text=f"Window level: {self.window_level}")
        self.window_width_text.configure(text=f"Window width: {self.window_width}")

        # Update window and DICOM viewer
        self.update_window()

    def upload_image_to_label(self, label_widget, image, mask=None, scale_factor=None): 
        """Uploading images to label widgets."""  
        # Ensure the input image is in uint8 format (0-255 range)
        if image.dtype != np.uint8:
            if image.max() <= 1.0: 
                image = (image * 255).astype(np.uint8)
            else: 
                image = (image / image.max() * 255).astype(np.uint8)

        # Apply upscaling if scale_factor is provided
        if scale_factor is not None and scale_factor > 1:
            # Calculate new dimensions
            new_height = int(image.shape[0] * scale_factor)
            new_width = int(image.shape[1] * scale_factor)
            # Use cv2.resize with INTER_LINEAR or INTER_CUBIC for better quality
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Convert grayscale to RGB if necessary
        if len(image.shape) == 2: 
            image_rgb = np.stack((image,) * 3, axis=-1)
        else:
            image_rgb = image

        if mask is not None and np.min(mask) == 0:
            # Ensure mask is binary
            mask = (mask > 0).astype(np.uint8) 
            # Create a colored overlay (e.g., red with transparency)
            overlay = np.zeros_like(image_rgb, dtype=np.uint8)
            overlay[mask > 0] = [255, 0, 0] 
            # Blend the overlay with the original image
            alpha = 0.25 
            image_with_mask = (image_rgb * (1 - alpha) + overlay * alpha).astype(np.uint8)
            # Convert to PIL image
            image_with_mask = Image.fromarray(image_with_mask)
        else:
            # No mask, just use the RGB image
            image_with_mask = Image.fromarray(image_rgb)

        # Convert to Tkinter-compatible format
        image_tk = ImageTk.PhotoImage(image_with_mask)

        # Update the label widget
        label_widget.config(image=image_tk)
        label_widget.image = image_tk 

    def generate_radiograph(self):
        """Generate radiographs and update the radiograph viewer."""
        # Rotate the CT volume 
        if self.sagittal_radiograph_angle != 0 or self.coronal_radiograph_angle != 0 or self.axial_radiograph_angle != 0:
            radiograph = rotate_3D(
                self.dicom_for_radiograph, 
                sagittal_angle=self.sagittal_radiograph_angle,
                coronal_angle=self.coronal_radiograph_angle,
                axial_angle=self.axial_radiograph_angle
            )
        else:
            radiograph = self.dicom_for_radiograph

        # Generate radiographs
        if self.style_radiograph == self.radiograph_styles[0]:
            self.radiograph_sagittal = compress_bonemri(radiograph, axis=0)
            self.radiograph_axial = compress_bonemri(radiograph, axis=1)
            self.radiograph_coronal = compress_bonemri(radiograph, axis=2)
        else:
            self.radiograph_sagittal = get_radiograph(radiograph, axis=2)
            self.radiograph_axial = get_radiograph(radiograph, axis=1)
            self.radiograph_coronal = get_radiograph(radiograph, axis=0)

        # Update radiograph viewer
        self.upload_image_to_label(
            self.sagittal_radiograph_image_label, 
            self.radiograph_sagittal
        )
        self.upload_image_to_label(
            self.axial_radiograph_image_label, 
            np.rot90(self.radiograph_axial, 3)
        )
        self.upload_image_to_label(
            self.coronal_radiograph_image_label, 
            np.rot90(self.radiograph_coronal, 3)
        )

        # Update vertebra viewer
        if self.target_vertebra is not None:
            self.update_bbox()
            self.vertebra_sag_angle_text_label.configure(
                text=f"Current sagittal angle: {self.sagittal_radiograph_angle:.1f} degrees"
            )
            self.vertebra_ax_angle_text_label.configure(
                text=f"Current axial angle: {self.axial_radiograph_angle:.1f} degrees"
            )

    def radiograph_select(self, choice):
        """Selection of full/segmented radiograph."""
        # Generate correct radiograph
        if choice == self.segment_radiograph[0]:
            self.dicom_for_radiograph = self.window_dicom_image
        else:
            self.dicom_for_radiograph = self.segment_dicom_image

    def radiograph_select_style(self, choice):
        """Selection of style radiograph."""
        # Generate correct radiograph
        if choice == self.radiograph_styles[0]:
            self.style_radiograph = self.radiograph_styles[0]
        else:
            self.style_radiograph = self.radiograph_styles[1]

    def update_radiograph_angle_sagittal(self, value):
        """Update the displayed angle based on the slider value."""
        self.sagittal_radiograph_angle = value
        self.sagittal_radiograph_slider_text.configure(
            text=f"Sagittal angle: {self.sagittal_radiograph_angle:.1f} degrees"
        )
    
    def update_radiograph_angle_coronal(self, value):
        """Update the displayed angle based on the slider value."""
        self.coronal_radiograph_angle = value
        self.coronal_radiograph_slider_text.configure(
            text=f"Coronal angle: {self.coronal_radiograph_angle:.1f} degrees"
        )
    
    def update_radiograph_angle_axial(self, value):
        """Update the displayed angle based on the slider value."""
        self.axial_radiograph_angle = value
        self.axial_radiograph_slider_text.configure(
            text=f"Axial angle: {self.axial_radiograph_angle:.1f} degrees"
        )
    
    def vertebra_select(self, choice):
        """Selection of target vertebra and update DICOM viewer."""
        # Translate vertebra to label
        vertebra_to_label = {"None":0, "T1":1, "T2":2, "T3":3, "T4":4, "T5":5, "T6":6, "T7":7, "T8":8, "T9":9, "T10":10, "T11":11, "T12":12, "L1":13, "L2":14, "L3":15, "L4":16, "L5":17}
        self.target_vertebra = vertebra_to_label[choice]
        self.vertebra_mask = (self.segment_data == self.target_vertebra).astype(float)

        # Update DICOM viewer
        self.upload_image_to_label(
            self.sagittal_dicom_image_label, 
            self.window_dicom_image[self.sagittal_slice_idx,:,:],
            self.vertebra_mask[self.sagittal_slice_idx,:,:]
        )
        self.upload_image_to_label(
            self.axial_dicom_image_label, 
            np.rot90(self.window_dicom_image[:,self.axial_slice_idx,:], 3),
            np.rot90(self.vertebra_mask[:,self.axial_slice_idx,:], 3)
        )
        self.upload_image_to_label(
            self.coronal_dicom_image_label, 
            np.rot90(self.window_dicom_image[:,:,self.coronal_slice_idx], 3),
            np.rot90(self.vertebra_mask[:,:,self.coronal_slice_idx], 3)
        )

        # Activate angle calculation buttons
        self.reference_trueAP_button.configure(state="normal")
        self.random_radiograph_button.configure(state="normal")
        self.optimize_angles_button.configure(state="normal")

        # Remove optimized bbox image
        self.optimized_bbox_label.config(image=None)
        self.optimized_bbox_label.image = None 

        self.opt_sag_angle_text_label.configure(
            text=f"Optimized sagittal angle: 0 degrees"
        )
        self.opt_ax_angle_text_label.configure(
            text=f"Optimized axial angle: 0 degrees"
        )

    def calculate_reference_trueAP_angles(self):
        """Updates radiograph viewer with reference angles for target vertebra."""
        # Obtain euler angles for target vertebra
        if self.target_vertebra != 0:
            self.target_angles = self.euler_angles[self.target_vertebra]
            self.sagittal_radiograph_angle = self.target_angles[0]
            self.axial_radiograph_angle = self.target_angles[2]

            # Update radiograph viewer
            self.generate_radiograph()
            self.sagittal_radiograph_slider.set(self.sagittal_radiograph_angle)
            self.sagittal_radiograph_slider_text.configure(
                text=f"Sagittal angle: {self.sagittal_radiograph_angle:.1f} degrees"
            )
            self.axial_radiograph_slider.set(self.axial_radiograph_angle) 
            self.axial_radiograph_slider_text.configure(
                text=f"Axial angle: {self.axial_radiograph_angle:.1f} degrees"
            )

            # Update vertebra viewer
            self.vertebra_sag_angle_text_label.configure(
                text=f"Current sagittal angle: {self.sagittal_radiograph_angle:.1f} degrees"
            )
            self.vertebra_ax_angle_text_label.configure(
                text=f"Current axial angle: {self.axial_radiograph_angle:.1f} degrees"
            )
            self.optimized_bbox_label.config(image=None)
            self.optimized_bbox_label.image = None 

            self.opt_sag_angle_text_label.configure(
                text=f"Optimized sagittal angle: 0 degrees"
            )
            self.opt_ax_angle_text_label.configure(
                text=f"Optimized axial angle: 0 degrees"
            )

    def generate_random_angles(self):
        """Updates radiograph viewer with random angles."""
        # Obtain random angles
        self.sagittal_radiograph_angle = random.uniform(-45, 45)
        self.axial_radiograph_angle = random.uniform(-10, 10)

        # Update radiograph viewer
        self.generate_radiograph()
        self.sagittal_radiograph_slider.set(self.sagittal_radiograph_angle)
        self.sagittal_radiograph_slider_text.configure(
            text=f"Sagittal angle: {self.sagittal_radiograph_angle:.1f} degrees"
        )
        self.axial_radiograph_slider.set(self.axial_radiograph_angle) 
        self.axial_radiograph_slider_text.configure(
            text=f"Axial angle: {self.axial_radiograph_angle:.1f} degrees"
        )

        # Update vertebra viewer
        self.vertebra_sag_angle_text_label.configure(
            text=f"Current sagittal angle: {self.sagittal_radiograph_angle:.1f} degrees"
        )
        self.vertebra_ax_angle_text_label.configure(
            text=f"Current axial angle: {self.axial_radiograph_angle:.1f} degrees"
        )
        self.optimized_bbox_label.config(image=None)
        self.optimized_bbox_label.image = None 

    def update_bbox(self):
        """Update the vertebra viewer."""
        # Apply bbox to get target vertebra 3D image
        rot_segment_dicom_image = rotate_3D(self.segment_dicom_image,
            sagittal_angle=self.sagittal_radiograph_angle,
            axial_angle=self.axial_radiograph_angle
        )
        rot_vertebra_mask = rotate_3D(self.vertebra_mask,
            sagittal_angle=self.sagittal_radiograph_angle,
            axial_angle=self.axial_radiograph_angle
        )
        rot_bbox_coord, rot_bbox_mask = compute_bbox(rot_vertebra_mask)
        rot_bbox_vertebra_image = rot_segment_dicom_image[
            rot_bbox_coord["z_min"]-5:rot_bbox_coord["z_max"]+5,
            rot_bbox_coord["y_min"]:rot_bbox_coord["y_max"],
            rot_bbox_coord["x_min"]:rot_bbox_coord["x_max"],
        ]
        if self.style_radiograph == self.radiograph_styles[0]:
            radiograph_bbox = compress_bonemri(rot_bbox_vertebra_image, axis=2)
        else:
            radiograph_bbox = get_radiograph(rot_bbox_vertebra_image, axis=0)

        # Update vertebra viewer
        self.upload_image_to_label(self.radiograph_bbox_label, 
            image=np.rot90(radiograph_bbox, 3),
            scale_factor=7
        )

    def optimize_angles(self):
        """Start angle optimization algorithm on current bbox."""
        # Rotate image and mask with angles currently active in GUI
        rot_segment_dicom_image = rotate_3D(self.segment_dicom_image,
            sagittal_angle=self.sagittal_radiograph_angle,
            axial_angle=self.axial_radiograph_angle
        )
        rot_vertebra_mask = rotate_3D(self.vertebra_mask,
            sagittal_angle=self.sagittal_radiograph_angle,
            axial_angle=self.axial_radiograph_angle
        )

        # Obtain bbox coord of rotated vertebra
        bbox_coord, _ = compute_bbox(rot_vertebra_mask)
        rot_vertebra_bbox_image = rot_segment_dicom_image[
            bbox_coord["z_min"]-5:bbox_coord["z_max"]+5,
            bbox_coord["y_min"]:bbox_coord["y_max"],
            bbox_coord["x_min"]:bbox_coord["x_max"]
        ]
        # Generate template image
        template = generate_template(self.segment_dicom_image, 
            self.segment_data, 
            self.target_vertebra, 
            self.euler_angles
        )
        # Bounds the saggital, coronal, and axial angles
        bounds = [(-45, 45), (0, 0), (-15, 15)] 

        # Optimize the rotation angles
        optimal_angles = optimize_rotation(rot_vertebra_bbox_image, template, bounds)

        # Apply optimized rotation angles to rotated full image and mask
        optimized_segment_dicom_image = rotate_3D(rot_segment_dicom_image,
            sagittal_angle=optimal_angles[0],
            axial_angle=optimal_angles[2]
        )
        optimized_vertebra_mask = rotate_3D(rot_vertebra_mask,
            sagittal_angle=optimal_angles[0],
            axial_angle=optimal_angles[2]
        )

        # Find bbox of the image rotated by the optimized angles
        optimized_bbox_coord, _ = compute_bbox(optimized_vertebra_mask)
        optimized_bbox_vertebra_image = optimized_segment_dicom_image[
            optimized_bbox_coord["z_min"]-5:optimized_bbox_coord["z_max"]+5,
            optimized_bbox_coord["y_min"]:optimized_bbox_coord["y_max"],
            optimized_bbox_coord["x_min"]:optimized_bbox_coord["x_max"]
        ]
        if self.style_radiograph == self.radiograph_styles[0]:
            optimized_radiograph_bbox = compress_bonemri(optimized_bbox_vertebra_image, axis=2)
        else:
            optimized_radiograph_bbox = get_radiograph(optimized_bbox_vertebra_image, axis=0)

        # Update vertebra viewer
        self.upload_image_to_label(self.optimized_bbox_label, 
            image=np.rot90(optimized_radiograph_bbox, 3),
            scale_factor=7
        )

        # Update text in viewer
        self.opt_sag_angle_text_label.configure(
            text=f"Optimized sagittal angle: {optimal_angles[0]:.1f} degrees"
        )
        self.opt_ax_angle_text_label.configure(
            text=f"Optimized axial angle: {optimal_angles[2]:.1f} degrees"
        )
