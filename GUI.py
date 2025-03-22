import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import cv2
import os
import io

import tkinter as tk
import tkinter.messagebox
import customtkinter as ctk

from tkinter import filedialog
from PIL import Image, ImageTk

from image_utils import *

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

        # RIGHT SIDE BAR ---------------------------------------------------------------------------------------------------------
        self.right_tabview = ctk.CTkTabview(self, width=300)
        self.right_tabview.grid(row=0, column=3, padx=(20, 10), pady=(20, 0), sticky="nsew")

        self.right_tabview.add("Vertebra")
        self.vertebra_frame = self.right_tabview.tab("Vertebra")
        self.vertebra_frame.grid_columnconfigure(0, weight=1) 
        self.vertebra_frame.grid_rowconfigure((0, 1), weight=0) 

        # Text label
        self.target_vertebra_text = ctk.CTkLabel(self.vertebra_frame,
            text="Select target vertebra"
        )
        self.target_vertebra_text.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

        # Option menu for target vertebra
        self.vertebra_mask = None
        self.vertebra = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
        self.optionmenu = ctk.CTkOptionMenu(self.vertebra_frame,
            values=self.vertebra,
            command=self.vertebra_select
        )
        self.optionmenu.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")


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
        self.dicom_image = apply_window(self.dicom_image, window_level=450, window_width=1500)

        # Load segmentation data
        segment = sitk.ReadImage(self.segmentation_path)
        self.segment_data = sitk.GetArrayFromImage(segment)

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

    def update_slice_sagittal(self, value):
        """Update the displayed slice based on the slider value."""
        self.sagittal_slice_idx = int(float(value))
        self.upload_image_to_label(
            self.sagittal_dicom_image_label, 
            self.dicom_image[self.sagittal_slice_idx, :, :]
        )
        self.sagittal_slider_text.configure(text=f"Sagittal slice {self.sagittal_slice_idx}")

    def update_slice_axial(self, value):
        """Update the displayed slice based on the slider value."""
        self.axial_slice_idx = int(float(value))
        self.upload_image_to_label(
            self.axial_dicom_image_label, 
            np.rot90(self.dicom_image[:, self.axial_slice_idx, :], 3)
        )
        self.axial_slider_text.configure(text=f"Axial slice {self.axial_slice_idx}")

    def update_slice_coronal(self, value):
        """Update the displayed slice based on the slider value."""
        self.coronal_slice_idx = int(float(value))
        self.upload_image_to_label(
            self.coronal_dicom_image_label, 
            np.rot90(self.dicom_image[:, :, self.coronal_slice_idx], 3)
        )
        self.coronal_slider_text.configure(text=f"Coronal slice {self.coronal_slice_idx}")

    def upload_image_to_label(self, label, image): 
        """Uploading images to label widgets."""      
        # Prepare image object
        if isinstance(image, np.ndarray):
            # Convert NumPy array to PIL Image
            image_PIL = Image.fromarray(image)
        elif isinstance(image, plt.Figure):
            # Convert matplotlib figure to PIL Image
            buf = io.BytesIO()
            image.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            image_PIL = Image.open(buf)
        
        # Turn image into photo and place in widget
        image_tk = ImageTk.PhotoImage(image_PIL)
        label.config(image=image_tk)
        label.image = image_tk

    def vertebra_select(self, choice):
        """Selection of target vertebra and update DICOM viewer."""
        # Translate vertebra to label
        vertebra_to_label = {"T1":1, "T2":2, "T3":3, "T4":4, "T5":5, "T6":6, "T7":7, "T8":8, "T9":9, "T10":10, "T11":11, "T12":12, "L1":13, "L2":14, "L3":15, "L4":16, "L5":17}
        self.target_vertebra = vertebra_to_label[choice]

        # Update DICOM viewer
        self.vertebra_mask = (self.segment_data == self.target_vertebra)

    def generate_contour_plot(self, segmentation_slice):
        """Generate a contour plot of the segmentation slice using matplotlib."""
        fig = plt.figure()
        plt.contour(segmentation_slice)
        plt.axis('off')
        return fig
