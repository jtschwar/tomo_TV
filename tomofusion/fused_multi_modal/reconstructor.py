from tomo_tv.fused_multi_modal.Utils import utils_cs_eds as utils
from tomo_tv.fused_multi_modal.Utils import mm_astra
import matplotlib.pyplot as plt
from typing import Dict
from tqdm import tqdm
import numpy as np

class reconstructor:

    def __init__(self, 
            haadf: np.ndarray, haadfTiltAngles: np.ndarray, 
            chem: Dict, chemTiltAngles: np.ndarray, 
            gamma: float = 1.6, sigmaMethod: int = 3):
        """
        
        """

        # Make Sure Haadf Projections are Non-Negative
        haadf[haadf<0] = 0

        # Get Necessary MetaData
        (self.nx, self.ny, _) = haadf.shape
        nPix = self.nx * self.ny

        # Get Meta Data from Chemical Dictionary
        elements = list(chem)
        self.nz = len(chem)

        # Initialize Tomography Operator
        self.tomo = mm_astra.mm_astra(
            self.nx, self.ny, self.nz, 
            np.deg2rad(haadfTiltAngles),
            np.deg2rad(chemTiltAngles))
        
        # Pass the Data to the Reconstructor
        self.set_haadf_projections( haadf, haadfTiltAngles)
        self.set_chemical_projections( chem, chemTiltAngles)
        self.set_summation_matrix( gamma, sigmaMethod)

        # Initialize Projection Operators
        self.tomo.initialize_FP(); self.tomo.initialize_BP(); self.tomo.initialize_SIRT()

        # Estimate Lipschitz
        self.tomo.estimate_lipschitz()

        # Bool for Measuring Cost Function
        self.tomo.set_measureChem(True); self.tomo.set_measureHaadf(True)

        self.reconTotal = np.zeros([self.nz,self.nx,self.ny,self.ny],dtype=np.float32)

    def set_haadf_projections(self, haadf, haadf_tilt_angles):

        # Pass HAADF Projections
        self.NprojHAADF = haadf_tilt_angles.shape[0]
        bh = np.zeros([self.nx,self.ny*self.NprojHAADF])
        for s in range(self.nx):
            bh[s,] = haadf[s,].transpose().flatten()
        self.tomo.set_haadf_tilt_series(bh)

    def set_chemical_projections(self, chem, chem_tilt_angles):

        # Pass Chemical Projections
        Nproj
        bChem = np.zeros([self.nx,self.nx*NprojCHEM*self.nz],dtype=Ti.dtype)
        for ss in range(self.nx):
            bChem[ss,] = np.concatenate([Ti[ss,].T.flatten(), C[ss,].T.flatten()])
        self.tomo.set_chem_tilt_series(bChem)

    def update_summation_matrix(self, gamma: float = 1.6, sigmaMethod: int = 3):

        # define gamma 
        self.tomo.set_gamma(gamma)

        # Create Summation Matrix
        sigmaMethod = 3
        sigma = utils.create_weighted_summation_matrix(self.nx, self.nx, self.nz, zNum, 1.6,sigmaMethod)
        (rows, cols) = sigma.nonzero(); vals = sigma.data
        sigma = np.array([rows,cols,vals],dtype=np.float32,order='C')
        self.tomo.load_sigma(sigma)


    def chemical_tomography(self, Niter: int = 250, lambdaCHEM: float = 0.05, show_convergence: bool = True):
        """
        
        """

        # Make Sure We Are Starting with A Fresh Reconstruction 
        self.tomo.restart_recon()

        # Main Loop 
        costCHEM = np.zeros(Niter)
        for ii in tqdm(range(Niter)):
            costCHEM[ii] = self.tomo.poisson_ml(lambdaCHEM)

        # Show Convergene 
        if show_convergence:
            plt.figure(figsize=(10,3)); plt.plot(costCHEM)
            plt.xlim([0,Niter-1]); plt.xlabel('# Iterations'); plt.ylabel('Cost')
            
    def data_fusion(self, 
            Niter:int = 50, lambdaTV: float = 1e-4, lambdaCHEM: float = 5e-2, 
            lambdaHAADF: float = 10, show_convergence: bool = True):
        """
        
        """

        # store parameters
        costCHEM = np.zeros(Niter, dtype=np.float32)
        costHAADF = costCHEM.copy(); costTV = costCHEM.copy()
        params = {'lambdaTV':lambdaTV, 'tvIter':self.tvIter, 'Niter':Niter, 'gamma': self.gamma,
                'lambdaCHEM':lambdaCHEM, 'lambdaHAADF':lambdaHAADF, 'iterSIRT':self.iterSIRT,
                'sigmaMethod':self.sigmaMethod, 'reduceLambda':self.reduceLambda}

        # main loop
        for i in tqdm(range(Niter)):
            (costHAADF[i], costCHEM[i]) = tomo.sirt_data_fusion(lambdaHAADF,lambdaCHEM,iterSIRT)
            costTV[i] = tomo.tv_fgp_4D(tvIter,lambdaTV)
            if i > 0 and costHAADF[i] > costHAADF[i-1]: lambdaCHEM *= 0.95

        # show cost function
        if show_convergence:
            plt.figure(figsize=(25,3))

            ax1 = plt.subplot(1,3,1); ax1.set_ylabel(r'$||A (\Sigma x) - b||^2$')
            ax2 = plt.subplot(1,3,2); ax2.set_ylabel(r'$\sum (Ax - b \cdot \log(Ax))$')
            ax3 = plt.subplot(1,3,3); ax3.set_ylabel(r'$\sum \|x\|_{TV}$')
            ax1.plot(costHAADF); ax2.plot(costCHEM); ax3.plot(costTV)


    def get_reconstruction(self):
        for e in range(self.nz):
            for s in range(self.nx):
                self.reconTotal[e,s,] = self.tomo.get_recon(e,s)
        return self.reconTotal

    # Auxilary Function to Visualize Slices of the Phantom Object or Reconstruction
    def display_recon_slices(self, slice, delta):                                                                                                                                                                                                                                                                                                                                                                                      
        fig, ax = plt.subplots(1,2,figsize=(12,20))
        ax = ax.flatten()

        for i in range(len(self.elements)):
            ax[i].imshow(np.mean(self.reconTotal[i,:,slice-int(delta/2):slice+int(delta/2),:],axis=1),cmap='gray'); 
            ax[i].set_title(self.elements[i]+': '+str(np.min(self.reconTotal[i,]))+' '+str(np.max(self.reconTotal[i,]))); ax[i].axis('off')