from tomofusion.gpu.utils import tomoengine, pytvlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

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

        plt.plot(cost)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title(f'{algorithm} Convergence')
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

    # Auxilary Function to Visualize Slices of the Phantom Object or Reconstruction
    def show_recon(self):
        """ Show the Reconstruction Slices in 3D """

        # Get Reconsturction is not done yet
        if self.recon is None:
            self.get_recon()

        (nx, ny, nz) = self.recon.shape
        fig, ax = plt.subplots(1,3,figsize=(25,25))
        ax = ax.flatten()
        ax[0].imshow(self.recon[int(nx/2),],cmap='gray'); ax[0].axis('off')
        ax[1].imshow(self.recon[:,int(ny/2),:],cmap='gray'); ax[1].axis('off')
        ax[2].imshow(self.recon[:,:,int(nz/2)],cmap='gray'); ax[2].axis('off')
        plt.show()

