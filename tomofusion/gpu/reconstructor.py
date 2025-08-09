from tomofusion.gpu.utils import tomoengine, pytvlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk

class TomoGPU:

    def __init__(self, tiltAngles: np.ndarray, tiltSeries: np.ndarray = None):
        """ Initialize the Reconstructor with Tilt Angles and Series"""

        # Check if GPU Support is Available
        pytvlib.check_cuda()

        # Initialize the C++ Object..
        self.Nslice, self.Nray, self.Nangles = tiltSeries.shape
        self.tomo = tomoengine(self.Nslice, self.Nray, np.deg2rad(tiltAngles))
        self.set_tilt_series(tiltSeries)

        # Null Volume Until Reconstruction is Complete
        self.recon = None

    def set_tilt_series(self, tiltSeries: np.ndarray):
        """ Set the Tilt Series for Reconstruction """

        self.Nslice, self.Nray, self.Nangles = tiltSeries.shape

        # Initialize the C++ Object..
        if self.tomo is None:
            self.tomo = tomoengine(self.Nslice, self.Nray, np.deg2rad(tiltAngles))
            self.set_tilt_series(tiltSeries)

        # Null Volume Until Reconstruction is Complete
        self.recon = None

        b = np.zeros([self.Nslice, self.Nray*self.Nangles])
        for s in range(self.Nslice):
            b[s,:] = tiltSeries[s,:,:].transpose().ravel()
            
        # Pass the Tilt Series to tomo_TV C++ object.
        self.tomo.set_tilt_series(b)

    def wbp(self, filter: str = 'ram-lak'):
        """ Perform a Filtered Back Projection Reconstruction """

        # Check if the filter is supported
        if filter not in pytvlib.wbp_filters():
            print(f'{filter} Filter not supported. Defaulting to ram-lak.')
            filter = 'ram-lak'

        # Initialize the Algorithm
        pytvlib.initialize_algorithm(self.tomo,'FBP',filter)

        # Reconstruct the Data
        pytvlib.run(self.tomo,'FBP')

    def sart(self, Niter: int = 150, init: float = 'sequential',
             show_convergence: bool = True):
        """ Perform a Simultaneous Algebraic Reconstruction Technique (SART) Reconstruction """

        # Check if the initialization method is supported
        if init not in pytvlib.sart_orders():
            print(f'{init} initialization method not supported. Defaulting to sequential.')
            init = 'sequential'
        
        # Initialize the Algorithm        
        pytvlib.initialize_algorithm(self.tomo,'SART',init)
        self._run_iterative('SART', Niter)

    def sirt(self, Niter: int = 150, show_convergence: bool = True):
        """ Perform a Simultaneous Iterative Reconstruction Technique (SIRT) Reconstruction """

        pytvlib.initialize_algorithm(self.tomo,'SIRT')
        self._run_iterative('SIRT', Niter)  

    def cgls(self, Niter: int = 100, show_convergence: bool = True):
        """ Perform a Conjugate Gradient Least Squares (CGLS) Reconstruction """

        pytvlib.initialize_algorithm(self.tomo,'CGLS')
        self._run_iterative('CGLS', Niter)             

    def _run_iterative(self, alg: str, Niter: int, show_convergence: bool = True):
        
        # Main Loop
        self.cost = np.zeros(Niter)
        self.tomo.restart_recon()
        for i in tqdm(range(Niter)):
            pytvlib.run(self.tomo,alg)
            if show_convergence:
                self.cost[i] = self.tomo.data_distance()
        
        if show_convergence:
            self.plot_convergence(self.cost, alg)

    def kl_divergence(self, Niter: int = 100, lambda_param: float = 0.1):
    
        self.tomo.restart_recon()
        print('Running Reconstruction...')
        pytvlib.initialize_algorithm(self.tomo,'kl-divergence')
        for i in tqdm(range(Niter)):
            pytvlib.run(self.tomo,'kl-divergence',lambda_param)

    def fista(self, 
        Niter: int = 100, momentum: bool = True, 
        lambda_param: float = 0.1, nTViter: int = 10,
        show_convergence: bool = True):
        """ Perform a Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) Reconstruction """

        pytvlib.initialize_algorithm(self.tomo,'fista')

        # (Optional): Ignore Momentum Acceleration
        if not momentum: self.tomo.remove_momentum()

        # Momentum and Objective Function 
        self.cost = np.zeros(Niter); t0 = 1

        # Main Loop 
        print('Running Reconstruction...')        
        for k in tqdm(range(Niter)):
            # Gradient Step
            pytvlib.run(self.tomo,'fista')
            
            # Threshold Step
            self.tomo.tv_fgp(nTViter,lambda_param)
            
            # Momentum Step
            if momentum: 
                tk = 0.5 * (1 + np.sqrt(1 + 4 * t0**2))
                self.tomo.fista_momentum((t0-1)/tk)
                t0 = tk

            # Measure Objective  
            if show_convergence:
                self.cost[k] = 0.5 * self.tomo.data_distance()**2 + lambda_param * self.tomo.tv()

        if show_convergence:
            self.plot_convergence(self.cost, 'FISTA')

    def asd_pocs(self, 
        Niter: int = 100,    eps: float = 0.025, 
        beta0: float = 0.25, beta_reduce: float = 0.9985,
        r_max: float = 0.95, nTViter: int = 10,
        alpha: float = 0.2, alpha_reduce: float = 0.95,
        show_convergence: bool = True):

        #Main Loop
        beta = beta0
        dd_vec, tv_vec = np.zeros(Niter), np.zeros(Niter)
        print('Running Reconstruction...')
        for i in tqdm(range(Niter)): 
            
            self.tomo.copy_recon() 

            pytvlib.run(self.tomo, 'sart', beta) # Reconstruction
            beta *= beta_red # Beta Reduction

            # Measure Magnitude for TV - GD
            if (i == 0):
                dPOCS = self.tomo.matrix_2norm() * alpha
                dp = dPOCS / alpha
            else:
                dp = self.tomo.matrix_2norm()

            # Measure difference between exp / sim projections. 
            dd_vec[i] = self.tomo.data_distance()

            self.tomo.copy_recon()

            # TV Minimization
            tv_vec[i] = self.tomo.tv_gd(nTViter, dPOCS)
            dg = self.tomo.matrix_2norm()

            if (dg > dp * r_max and dd_vec[i] > eps):
                dPOCS *= alpha_red

    def plot_convergence(self, cost, algorithm):
        """ Plot the Convergence of the Reconstruction Algorithm """

        Niter = cost.shape[0]
        plt.figure(figsize=(8,5))
        plt.scatter(np.arange(Niter), cost)
        plt.xlabel('Iteration'); plt.ylabel('Cost')
        plt.title(f'{algorithm} Convergence')
        plt.xlim([0, Niter-1])
        plt.tick_params(direction='in', length=6, width=1.5, 
                        which='both', top=True, right=True)        
        plt.show()

    def get_recon(self):
        """ Get the Reconstruction from the C++ Reconstruction Object """

        # Return the Reconstruction to Python
        if self.recon is None:
            self.recon = np.zeros([self.Nslice, self.Nray, self.Nray])
        for s in range(self.Nslice):
            self.recon[s,] = self.tomo.get_recon(s)
        return self.recon
    
    def get_projections(self):

        pass

    def show_recon(self):
        """
        Interactive Tkinter viewer for 3D reconstruction
        Shows XY, XZ, YZ slices with sliders
        """
        # Get reconstruction if not done yet
        if self.recon is None:
            self.get_recon()
        
        nx, ny, nz = self.recon.shape
        
        root = tk.Tk()
        root.title("3D Reconstruction Viewer")
        root.geometry("1000x400")
        
        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Slice variables
        xy_slice = tk.IntVar(value=int(nx/2))
        xz_slice = tk.IntVar(value=int(ny/2)) 
        yz_slice = tk.IntVar(value=int(nz/2))
        
        # XY slice control
        ttk.Label(controls_frame, text="XY Slice:").grid(row=0, column=0, sticky=tk.W)
        xy_scale = ttk.Scale(controls_frame, from_=0, to=nx-1, variable=xy_slice, orient=tk.HORIZONTAL)
        xy_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        xy_label = ttk.Label(controls_frame, text=str(int(nx/2)))
        xy_label.grid(row=0, column=2, padx=5)
        
        # XZ slice control  
        ttk.Label(controls_frame, text="XZ Slice:").grid(row=1, column=0, sticky=tk.W)
        xz_scale = ttk.Scale(controls_frame, from_=0, to=ny-1, variable=xz_slice, orient=tk.HORIZONTAL)
        xz_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        xz_label = ttk.Label(controls_frame, text=str(int(ny/2)))
        xz_label.grid(row=1, column=2, padx=5)
        
        # YZ slice control
        ttk.Label(controls_frame, text="YZ Slice:").grid(row=2, column=0, sticky=tk.W)
        yz_scale = ttk.Scale(controls_frame, from_=0, to=nz-1, variable=yz_slice, orient=tk.HORIZONTAL)
        yz_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        yz_label = ttk.Label(controls_frame, text=str(int(nz/2)))
        yz_label.grid(row=2, column=2, padx=5)
        
        # Configure column weights
        controls_frame.columnconfigure(1, weight=1)
        
        # Images frame
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(fill=tk.BOTH, expand=True)
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1) 
        images_frame.columnconfigure(2, weight=1)
        images_frame.rowconfigure(0, weight=1)
        
        # Image labels
        xy_img_label = ttk.Label(images_frame)
        xy_img_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        xz_img_label = ttk.Label(images_frame)  
        xz_img_label.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        yz_img_label = ttk.Label(images_frame)
        yz_img_label.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        
        # Titles
        ttk.Label(images_frame, text="XY View", font=('Arial', 10, 'bold')).grid(row=1, column=0)
        ttk.Label(images_frame, text="XZ View", font=('Arial', 10, 'bold')).grid(row=1, column=1)  
        ttk.Label(images_frame, text="YZ View", font=('Arial', 10, 'bold')).grid(row=1, column=2)
        
        def normalize_image(img_data):
            """Normalize image data to 0-255 range"""
            img_min, img_max = img_data.min(), img_data.max()
            if img_max > img_min:
                return ((img_data - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                return np.zeros_like(img_data, dtype=np.uint8)
        
        def update_images():
            """Update all three slice views"""
            # Get current slice indices
            xy_idx = int(xy_slice.get())
            xz_idx = int(xz_slice.get()) 
            yz_idx = int(yz_slice.get())
            
            # Update labels
            xy_label.config(text=str(xy_idx))
            xz_label.config(text=str(xz_idx))
            yz_label.config(text=str(yz_idx))
            
            # Get slice data
            xy_data = self.recon[xy_idx, :, :]  # XY plane at z=xy_idx
            xz_data = self.recon[:, xz_idx, :]  # XZ plane at y=xz_idx  
            yz_data = self.recon[:, :, yz_idx]  # YZ plane at x=yz_idx
            
            # Normalize and convert to PIL
            display_size = (250, 250)
            
            # XY image
            xy_norm = normalize_image(xy_data)
            xy_pil = Image.fromarray(xy_norm, mode='L')
            xy_pil = xy_pil.resize(display_size, Image.Resampling.NEAREST)
            xy_photo = ImageTk.PhotoImage(xy_pil)
            xy_img_label.configure(image=xy_photo)
            xy_img_label.image = xy_photo
            
            # XZ image  
            xz_norm = normalize_image(xz_data)
            xz_pil = Image.fromarray(xz_norm, mode='L')
            xz_pil = xz_pil.resize(display_size, Image.Resampling.NEAREST)
            xz_photo = ImageTk.PhotoImage(xz_pil)
            xz_img_label.configure(image=xz_photo)
            xz_img_label.image = xz_photo
            
            # YZ image
            yz_norm = normalize_image(yz_data)
            yz_pil = Image.fromarray(yz_norm, mode='L')
            yz_pil = yz_pil.resize(display_size, Image.Resampling.NEAREST) 
            yz_photo = ImageTk.PhotoImage(yz_pil)
            yz_img_label.configure(image=yz_photo)
            yz_img_label.image = yz_photo
        
        # Bind slider changes
        def on_change(*args):
            update_images()
        
        xy_slice.trace('w', on_change)
        xz_slice.trace('w', on_change)  
        yz_slice.trace('w', on_change)
        
        # Keyboard controls
        def on_key(event):
            if event.keysym == 'q':
                xy_slice.set(max(0, xy_slice.get() - 1))
            elif event.keysym == 'w':  
                xy_slice.set(min(nx-1, xy_slice.get() + 1))
            elif event.keysym == 'a':
                xz_slice.set(max(0, xz_slice.get() - 1))
            elif event.keysym == 's':
                xz_slice.set(min(ny-1, xz_slice.get() + 1))
            elif event.keysym == 'z':
                yz_slice.set(max(0, yz_slice.get() - 1))
            elif event.keysym == 'x':
                yz_slice.set(min(nz-1, yz_slice.get() + 1))
        
        root.bind('<Key>', on_key)
        root.focus_set()
        
        # Initial display
        update_images()
        
        # Add instructions
        instructions = ttk.Label(main_frame, 
                                text="Keyboard: Q/W (XY), A/S (XZ), Z/X (YZ)",
                                font=('Arial', 8))
        instructions.pack(pady=5)
        
        root.mainloop()

