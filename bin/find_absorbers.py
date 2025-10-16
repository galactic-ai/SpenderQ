#!/usr/bin/env python
import argparse
import os

import pickle 
import numpy as np
import torch

from spenderq import util as U
from spenderq import load_model

from qsoabsfind import absfinder
from qsoabsfind.config import load_constants

from qsoabsfind.absorberutils import (
    estimate_local_sigma_conv_array,
    median_selection_after_combining,
    remove_Mg_falsely_come_from_Fe_absorber,
    z_abs_from_same_metal_absorber,
    contiguous_pixel_remover,
    estimate_snr_for_lines,
    absorber_search_window,
    find_valid_indices,
    calculate_doublet_ratio,
    group_and_select_weighted_redshift
)

constants = load_constants()
lines, oscillator_parameters, speed_of_light = constants.lines, constants.oscillator_parameters, constants.speed_of_light


def load_spenderq_model(name, nmodel=5):
    ''' load spenderq models
    '''
    _dir = '/tigress/chhahn/spender_qso/models/'

    models = [] 
    for imodel in range(nmodel): 
        # load model
        _models, losses = load_model(os.path.join(_dir, '%s.%i.pt' % (name, imodel)))
        models.append(_models[0])
    return models


def load_spectra(name, ibatch):  
    ''' load spectra 
    '''
    _dir = '/tigress/chhahn/spender_qso/train/'

    # load batch
    with open(os.path.join(_dir, '%s_%i.pkl' % (name, ibatch)), 'rb') as f: 
        spec, w, z, target_id, norm, zerr = pickle.load(f)

    return spec, w, z, target_id, norm, zerr


def get_recon(models, spec): 
    ''' get spenderq reconstructions
    '''
    recons = [] 
    for imodel in range(len(models)): 
        with torch.no_grad():
            models[imodel].eval()

            s = models[imodel].encode(spec)
            _recon = np.array(models[imodel].decode(s))

        recons.append(_recon)
    recon = np.mean(recons, axis=0)
    return recon 


def find_absorber(lam_obs, residual, w, z_qso, verbose=False, absorber='MgII'):
    constants = load_constants()
    kwargs = constants.search_parameters[absorber].copy()
    kwargs['logwave'] = False
    
    # Define the wavelength range for searching the absorber
    min_wave, max_wave = lam_obs.min() + kwargs["lam_edge_sep"], lam_obs.max() - kwargs["lam_edge_sep"]  # avoiding edges

    # Retrieve flux and error data, ensuring consistent dtype for Numba compatibility
    residual = residual.astype('float64')
    error = (w**-0.5).astype('float64')
    lam_obs = lam_obs.astype('float64')

    # Remove NaN values from the arrays
    non_nan_indices = ~np.isnan(residual) & ~np.isnan(error)
    lam_obs, residual, error = lam_obs[non_nan_indices], residual[non_nan_indices], error[non_nan_indices]
    
    # Identify the wavelength region for searching the specified absorber
    lam_search, unmsk_residual, unmsk_error = absorber_search_window(
        lam_obs, residual, error, z_qso, absorber, min_wave, max_wave, verbose=kwargs['verbose'])
    
    # Verify that the arrays are of equal size
    assert lam_search.size == unmsk_residual.size == unmsk_error.size, "Mismatch in array sizes of lam_search, unmsk_residual, and unmsk_error"

    kwargs.pop("lam_edge_sep") # just remove this keyword as its not used the following function.
    spec_index = 0 
    
    (index_spec, pure_z_abs, pure_gauss_fit, pure_gauss_fit_std, pure_ew_first_line_mean, pure_ew_second_line_mean, pure_ew_total_mean, pure_ew_first_line_error, pure_ew_second_line_error, pure_ew_total_error, redshift_err, sn1_all, sn2_all, vel_disp1, vel_disp2) = absfinder.convolution_method_absorber_finder_in_QSO_spectra(spec_index, absorber, lam_obs, residual, error, lam_search, unmsk_residual, unmsk_error, **kwargs)

    return (index_spec, pure_z_abs, pure_gauss_fit, pure_gauss_fit_std, pure_ew_first_line_mean, pure_ew_second_line_mean, pure_ew_total_mean, pure_ew_first_line_error, pure_ew_second_line_error, pure_ew_total_error, redshift_err, sn1_all, sn2_all, vel_disp1, vel_disp2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="data file directory")
    parser.add_argument("absorber", help="absorber")
    parser.add_argument("-m", "--model", help='model tag', type=str, default=None)
    parser.add_argument("-nm", "--n_model", help='number of models', type=int, default=5)
    parser.add_argument("-ti", "--input_tag", help="input data tag", type=str, default='')
    parser.add_argument("-i", "--ibatch", help="batch number", type=int, default=None)
    args = parser.parse_args()
    
    # load model 
    models = load_spenderq_model(args.model, nmodel=args.n_model)

    # load spectra
    spec, w, redshifts, _, _, _  = load_spectra(args.input_tag, args.ibatch) 

    # load recon 
    recon = get_recon(models, spec)
        
    # wavelengths
    w_obs = np.array(models[0].wave_obs)
    w_recon = np.array(models[0].wave_rest) # reconstruciton rest-frame wavelength 

    # get f_absorp
    fabsorp = np.zeros((recon.shape[0], len(w_obs)))
    for igal in range(recon.shape[0]):
        try:
            recon_rebin = U.trapz_rebin(w_recon * (1+np.array(redshifts)[igal]), recon[igal], xnew=w_obs)
            fabsorp[igal,:] = np.array(spec)[igal] / recon_rebin
        except ValueError:
            print(redshifts[igal])

    abss =[]
    for igal in range(len(redshifts)):
        if np.sum(np.abs(fabsorp)) == 0: continue 
        
        out = find_absorber(w_obs, fabsorp[igal], np.array(w)[igal], np.array(redshifts)[igal], absorber=args.absorber)
        out = list(out)
        out[0] = igal 
        if out[1][0] != 0: 
            abss.append(out)
    
    with open(os.path.join(args.dir, '%s.%s_%i.pkl' % (args.absorber, args.input_tag, args.ibatch)), 'wb') as f: 
        pickle.dump(abss, f) 
