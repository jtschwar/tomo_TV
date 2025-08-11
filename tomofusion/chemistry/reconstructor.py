from tomofusion.chemistry.utils import utils_cs_eds as utils
from tomofusion.chemistry.utils import multimodal, multigpufusion
import matplotlib.pyplot as plt
from typing import Dict
from tqdm import tqdm
import numpy as np
import tomofusion

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ChemicalTomo:

    def __init__(self, 
            haadf: np.ndarray, haadfTiltAngles: np.ndarray, 
            chem: Dict, chemTiltAngles: np.ndarray, 
            gamma: float = 1.6, sigmaMethod: int = 3,
            gpu_id: int = -1):
        """
        Initialize the Chemical Tomography Reconstructor
        Args:
            haadf (np.ndarray): HAADF tilt series with shape (Nx, Ny, Nangles_haadf)
            haadfTiltAngles (np.ndarray): 1D array of HAADF tilt angles in degrees. Shape: (Nangles_haadf,).
            chem (Dict[str, np.ndarray]): Dictionary mapping element names to their corresponding 
                                        tilt series. Each tilt series should have shape 
                                        (Nx, Ny, Nangles_chem). Keys are element symbols 
                                        (e.g., 'C', 'Zn') and values are 3D numpy arrays.
            chemTiltAngles (np.ndarray): 1D array of chemical mapping tilt angles in degrees. 
                                        Shape should be (Nangles_chem,).
            gamma (float, optional): Defaults to 1.6.
            sigmaMethod (int, optional): Method for computing the summation matrix used in 
                                    data fusion.
        """

        # Get Necessary MetaData
        (self.nx, self.ny, _) = haadf.shape
        nPix = self.nx * self.ny

        # Get Meta Data from Chemical Dictionary
        self.elements = list(chem)
        self.nz = len(chem)

        self.tomo = multimodal(
            self.nx, self.ny, self.nz, 
            np.deg2rad(haadfTiltAngles),
            np.deg2rad(chemTiltAngles))

        # Initialize Tomography Operator
        config = tomofusion.determine_gpu_config(gpu_id)
        if config == 'singleconfig':
            print(f"Initializing single-GPU configuration")
            self.tomo = multimodal(
                self.nx, self.ny, self.nz, 
                np.deg2rad(haadfTiltAngles),
                np.deg2rad(chemTiltAngles))
        elif config == 'multigpu':
            print(f"Initializing multi-GPU configuration")
            self.tomo = multigpufusion.multigpufusion(
                self.nx, self.ny, self.nz, 
                np.deg2rad(haadfTiltAngles),
                np.deg2rad(chemTiltAngles))

        # Check to see if we want to set the GPU ID
        if gpu_id >= 0 and config == 'singleconfig':
            self.tomo.set_gpu(gpu_id)

        self.NprojHAADF = haadfTiltAngles.shape[0]
        self.NprojCHEM = chemTiltAngles.shape[0]

        # Pass the Data to the Reconstructor
        self.set_haadf_projections( haadf )
        self.set_chemical_projections( chem )
        self.set_summation_matrix( gamma, sigmaMethod )

        # Set Hyperparameters
        self.gamma = gamma
        self.sigmaMethod = sigmaMethod
        self.reduceLambda = True

        # Initialize Projection Operators
        self.tomo.initialize_FP(); self.tomo.initialize_BP(); self.tomo.initialize_SIRT()

        # Estimate Lipschitz
        self.tomo.estimate_lipschitz()

        # Bool for Measuring Cost Function
        self.tomo.set_measureChem(True); self.tomo.set_measureHaadf(True)

        # Initialize Reconstruction Volume
        self.reconTotal = None
        self.chemistry_reconstructed = False

    def set_haadf_projections(self, haadf):
        """
        Set haadf projections for the reconstruction.
        """

        # Normalize & Make Sure Haadf Projections are Non-Negative
        haadf[haadf<0] = 0
        haadf /= np.max(haadf)

        # Pass HAADF Projections
        bh = np.zeros([self.nx,self.ny*self.NprojHAADF])
        for s in range(self.nx):
            bh[s,] = haadf[s,].transpose().flatten()
        self.tomo.set_haadf_tilt_series(bh)

    def set_chemical_projections(self, chem):
        """
        Set chemical projections for all elements in the reconstruction.
        
        Args:
            chem (Dict[str, np.ndarray]): Dictionary mapping element names to their 
                                        tilt series arrays with shape (nx, ny, NprojCHEM)
        """

        # Normalize & Make Sure Chemical Projections are Non-Negative
        for element in self.elements:
            chem[element][chem[element] < 0] = 0
            chem[element] /= np.max(chem[element])

        # Pass Chemical Projections
        bChem = np.zeros([self.nx, self.nx*self.NprojCHEM*self.nz],dtype=np.float32)
        
        for ss in range(self.nx):
            # Collect all elements for this slice
            element_data = []
            for element in self.elements:
                # Get slice ss for this element, transpose and flatten
                element_slice = chem[element][ss,].T.flatten()
                element_data.append(element_slice)
            
            # Concatenate all elements for this slice
            bChem[ss,] = np.concatenate(element_data)

        self.tomo.set_chem_tilt_series(bChem)

    def set_summation_matrix(self, gamma: float = 1.6, sigmaMethod: int = 3):
        """
        Define the Summation Matrix for the Data Fusion Reconstruction
        """

        # define gamma 
        self.tomo.set_gamma(gamma)

        # Get the Z-Numbers from the Input Elements
        pt_table = utils.get_periodic_table()
        zNums = list(map(lambda x: pt_table[x.lower()], self.elements))

        # Create Summation Matrix
        sigma = utils.create_weighted_summation_matrix(self.nx, self.nx, self.nz, zNums, 1.6,sigmaMethod)
        (rows, cols) = sigma.nonzero(); vals = sigma.data
        sigma = np.array([rows,cols,vals],dtype=np.float32,order='C')
        self.tomo.load_sigma(sigma)

    def chemical_tomography(self, Niter: int = 100, lambdaCHEM: float = 0.05, alg='kldivergnce', show_convergence: bool = True):
        """
        Run the Chemical Tomography Reconstruction (Non-Data Fusion)
        """
        print('Reconstructing Chemical Tomograms (Non-Multimodal)...')

        # Make Sure We Are Starting with A Fresh Reconstruction 
        self.tomo.restart_recon()

        # Main Loop 
        costCHEM = np.zeros(Niter)
        for ii in tqdm(range(Niter)):
            costCHEM[ii] = self.tomo.poisson_ml(lambdaCHEM)

        # Show Convergence 
        if show_convergence:
            plt.figure(figsize=(8,4)); plt.scatter(np.arange(Niter),costCHEM)
            plt.xlim([0,Niter-1]); plt.xlabel('# Iterations'); plt.ylabel('Cost')
            plt.tick_params(direction='in', length=6, width=1.5, 
                            which='both', top=True, right=True)
            plt.show()

        # Flag to Indicate Chemistry is Reconstructed
        self.chemistry_reconstructed = True
            
    def data_fusion(self, 
            Niter:int = 50,  
            lambdaCHEM: float = 5e-2, lambdaHAADF: float = 10, lambdaTV: float = 1e-4, 
            iterSIRT: int = 5, tvIter: int = 5, show_convergence: bool = True):
        """
        Fuse the Reconstructions from the HAADF and Chemistry Modalities
        """

        # If the Chemistry is Not Initially Reconstructed, Run the Reconstruction
        if self.chemistry_reconstructed == False:
            self.chemical_tomography(lambdaCHEM = lambdaCHEM, show_convergence=show_convergence)

        # Ensure Data is Rescaled
        self._rescale_data() 

        # store parameters
        costCHEM = np.zeros(Niter, dtype=np.float32)
        costHAADF = costCHEM.copy(); costTV = costCHEM.copy()
        params = {'lambdaTV':lambdaTV, 'tvIter':tvIter, 'Niter':Niter, 'gamma': self.gamma,
                'lambdaCHEM':lambdaCHEM, 'lambdaHAADF':lambdaHAADF, 'iterSIRT':iterSIRT,
                'sigmaMethod':self.sigmaMethod, 'reduceLambda':self.reduceLambda}

        # main loop
        print('Reconstructing Fused Chemistry (Multimodal)...')        
        for i in tqdm(range(Niter)):
            (costHAADF[i], costCHEM[i]) = self.tomo.sirt_data_fusion(lambdaHAADF,lambdaCHEM,iterSIRT)
            costTV[i] = self.tomo.tv_fgp_4D(tvIter,lambdaTV)
            if i > 0 and costHAADF[i] > costHAADF[i-1]: lambdaCHEM *= 0.95

        # show cost function
        if show_convergence:
            plt.figure(figsize=(9,6))
            ax1 = plt.subplot(3,1,1); ax1.set_ylabel(r'$||A (\Sigma x) - b||^2$')
            ax2 = plt.subplot(3,1,2); ax2.set_ylabel(r'$\sum (Ax - b \cdot \log(Ax))$')
            ax3 = plt.subplot(3,1,3); ax3.set_ylabel(r'$\sum \|x\|_{TV}$')
            ax1.plot(costHAADF); ax2.plot(costCHEM); ax3.plot(costTV)
            ax1.set_xticklabels([]); ax2.set_xticklabels([])
            ax3.set_xlabel('# Iterations')
            # Set ticks inward for all axes
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim([0, Niter-1])
                ax.tick_params(direction='in', length=6, width=1.5, 
                               which='both', top=True, right=True)
            plt.show()

    def _rescale_data(self, scale: float = 10):
        """
        Rescale the tomograms by a given factor.
        """

        # Rescale the Tomograms by a Given Value
        self.tomo.rescale_tomograms(scale)

        # Rescale the Projections Shortly After
        self.tomo.rescale_projections()

    def get_reconstruction(self):
        """
        Get the current reconstruction.
        """
        if self.reconTotal is None:
            self.reconTotal = np.zeros([self.nz,self.nx,self.ny,self.ny],dtype=np.float32)

        # return recon to python
        for e in range(self.nz):
            for s in range(self.nx):
                self.reconTotal[e,s,] = self.tomo.get_recon(e,s)
        return self.reconTotal
    
    def display_recon(self):
        """
        Interactive Tkinter viewer for chemical tomography reconstruction
        Shows all elements side-by-side with single slice control
        """
        # Get the Reconstruction
        self.get_reconstruction()

        slice = int(self.ny // 2)
        
        root = tk.Tk()
        root.title("Chemical Reconstruction Viewer")
        root.geometry("1200x500")
        
        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Slice variables
        slice_var = tk.IntVar(value=slice)
        
        # Slice control
        ttk.Label(controls_frame, text="Slice:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        slice_scale = ttk.Scale(controls_frame, from_=0, 
                            to=self.reconTotal.shape[2]-1, 
                            variable=slice_var, orient=tk.HORIZONTAL)
        slice_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        slice_label = ttk.Label(controls_frame, text=str(slice))
        slice_label.grid(row=0, column=2, padx=5)
        
        # Configure column weights
        controls_frame.columnconfigure(1, weight=1)  # Slice slider gets all space
        
        # Images frame
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid for equal distribution of elements
        for i in range(self.nz):
            images_frame.columnconfigure(i, weight=1, uniform="cols")
        images_frame.rowconfigure(0, weight=1)
        images_frame.rowconfigure(1, weight=0)  # Row for titles
        
        # Store image labels and titles
        img_labels = []
        title_labels = []
        
        # Create frames for each element
        for i in range(self.nz):
            # Image label
            img_label = ttk.Label(images_frame)
            img_label.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            img_labels.append(img_label)
            
            # Title label
            title_label = ttk.Label(images_frame, text=self.elements[i], 
                                font=('Arial', 10, 'bold'))
            title_label.grid(row=1, column=i, pady=(5, 0))
            title_labels.append(title_label)
        
        def normalize_image(img_data):
            """Normalize image data to 0-255 range"""
            img_min, img_max = img_data.min(), img_data.max()
            if img_max > img_min:
                return ((img_data - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                return np.zeros_like(img_data, dtype=np.uint8)
        
        def update_images():
            """Update all element images"""
            current_slice = int(slice_var.get())
            
            # Update labels
            slice_label.config(text=str(current_slice))
            
            # Calculate display size based on number of elements
            base_size = 800 // self.nz  # Distribute available width
            display_size = (max(200, base_size), max(200, base_size))
            
            # Update each element image
            for i in range(self.nz):
                # Get single slice (no averaging)
                img_data = self.reconTotal[i, :, current_slice, :]
                
                # Normalize and convert to PIL
                img_norm = normalize_image(img_data)
                pil_img = Image.fromarray(img_norm, mode='L')
                pil_img = pil_img.resize(display_size, Image.Resampling.NEAREST)
                photo = ImageTk.PhotoImage(pil_img)
                
                # Update image
                img_labels[i].configure(image=photo)
                img_labels[i].image = photo  # Keep reference
                
                # Update title with min/max values
                img_min, img_max = img_data.min(), img_data.max()
                title_text = f"{self.elements[i]}: {img_min:.6f} {img_max:.6f}"
                title_labels[i].config(text=title_text)
        
        # Bind slider changes
        def on_change(*args):
            update_images()
        
        slice_var.trace('w', on_change)
        
        # Keyboard controls
        def on_key(event):
            current_slice = slice_var.get()
            
            if event.keysym == 'Left':
                new_val = max(0, current_slice - 1)
                slice_var.set(new_val)
            elif event.keysym == 'Right':
                new_val = min(self.reconTotal.shape[2]-1, current_slice + 1)
                slice_var.set(new_val)
        
        root.bind('<Key>', on_key)
        root.focus_set()
        
        # Initial display
        update_images()
        
        # Add instructions
        instructions = ttk.Label(main_frame, 
                                text="Keyboard: Left/Right arrows to change slice",
                                font=('Arial', 8))
        instructions.pack(pady=5)
        
        root.mainloop()