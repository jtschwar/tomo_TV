from tomo_tv.gpu_3D.Utils import astra_ctvlib, pytvlib
from tomo_tv.gpu_3D import reconstructor
from tqdm import tqdm
import numpy as np

class simulator(reconstructor):

    def __init__(self, volume: np.ndarray, tiltAngles: np.ndarray, snr: float 5):
        
        # Set the Original Voume
        self.volume = volume
        (self.Nslice, self.Nray, _) = volume.shape

        # Initialize the C++ Object..
        tomo = astra_ctvlib.astra_ctvlib(self.Nslice, self.Nray, np.deg2rad(tiltAngles))

        # astra_ctvlib by default creates one 3D volume for the reconstruction, 
        # any additional volumes needs to be externally intialized 
        # (this is to save memory consumption)
        tomo.initialize_initial_volume()

        # Optional: Apply Poisson Noise to Background Volume
        if np.min(volume) == 0:
            volume[volume == 0] = 1

        # Let's pass the volume from python to C++  
        for s in range(self.Nslice):
            tomo.set_original_volume(volume[s,:,:],s)

        # Now Let's Create the Projection Images
        tomo.create_projections()

        # Optional: Apply poisson noise to volume.
        snr = 5
        if snr > 0: tomo.poisson_noise(snr)

    def _run_iterative(self, alg: str, Niter: int, show_convergence: bool = True):
        
        # Main Loop
        self.rmse, self.cost = np.zeros(Niter), np.zeros(Niter)
        self.tomo.restart_recon()
        for i in tqdm(range(Niter)):
            pytvlib.run(self.tomo,alg)
            self.rmse[i] = self.tomo.rmse()
            if show_convergence:
                self.cost[i] = self.data_distance()
        
        if show_convergence:
            self.plot_convergence(self.cost)