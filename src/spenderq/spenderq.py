'''


module to interface with trained spenderq models 



'''
import os
import pickle
import numpy as np
import torch

from . import load_model
from . import util as U
from . import lyalpha as LyA


class SpenderQ(object):
    def __init__(self, name, sigma_lya=1.1, sigma_lyb=0.8, wave_lya=1215.67, wave_lyb=1026.):
        ''' class object for SpenderQ trained model 

        parameters 
        ==========
        name : str
            name of spenderq model 

        sigma_lya : float
            sigma clipping used in the LyA region. (default: 1.1) 

        sigma_lyb : float
            sigma clipping used in the LyB region. (default: 0.8)

        wave_lya : float 
            LyA wavelength (default: 1215.67)
        
        wave_lyb : float 
            Lyb wavelength (default: 1026.)

        '''
        if name not in ['qso.dr1.hiz']: 
            raise NotImplementedError

        self.name = name
        self.sigma_lya = sigma_lya
        self.sigma_lyb = sigma_lyb
        self.wave_lya = wave_lya
        self.wave_lyb = wave_lyb
    
        # load spenderq models
        self._load()


    def eval(self, spec, w, z): 
        ''' given input spectra, w, z apply the iterative SpenderQ framework 

        parameters
        ==========
        spec : torch.Tensor 
            Nspec x Nwave tensor of spectra 

        w : torch.Tensor
            Nspec x Nwave tensor of weights  

        z : torch.Tensor
            Nspec  tensor of galaxy redshifts  

        returns
        ======= 
        s : Nspec x Ndim
            final latent 

        recon : Nspec x Nwave_r
            QSO spectral reconstruction
        '''
        Nspec = spec.shape[0] # number of spectra
        
        # iterate through models
        for model in self.models: 

            with torch.no_grad():
                model[0].eval()
                
                s = model[0].encode(spec) # encode the spectra

                recon = np.array(model[0].decode(s)) # run decoder 

            for igal in np.arange(Nspec):
                # identify LyA absorption
                is_absorb = LyA.identify_absorp(
                    np.array(model[0].wave_obs),
                    np.array(spec[igal]),
                    np.array(w[igal]),
                    np.array(z)[igal],
                    np.array(model[0].wave_rest * (1 + z[igal])),
                    np.array(recon[igal]),
                    sigma_lya=self.sigma_lya,
                    sigma_lyb=self.sigma_lyb,
                    wave_lya=self.wave_lya,
                    wave_lyb=self.wave_lyb,
                    method='snr_rebin')

                # update weights
                w[igal,is_absorb] = 0.

        return s, recon 

    def wave_recon(self): 
        ''' reconstruction wavelength 
        '''
        return np.array(self.models[0][0].wave_rest)

    def _load(self): 
        ''' load all iterations of the spenderq models 
        '''
        # saved models 
        dir_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat')
        
        # check that all the models are there
        for str in ['', '.i0', '.i1', '.i2', '.i3']: 
            if not os.path.isfile(os.path.join(dir_model, '%s%s.pt' % (self.name, str))): 
                raise ValueError("model is missing") 


        # load the models
        models = [] 
        for str in ['', '.i0', '.i1', '.i2', '.i3']: 
            model, _ = load_model(os.path.join(dir_model, '%s%s.pt' %
                                               (self.name, str)))
            models.append(model) 
        
        self.models = models 
        return None 
