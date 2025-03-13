import pickle
import numpy as np
import glob
import torch
import math

from astropy.io import fits
from astropy.table import Table

from spenderq import load_model
from spenderq import util as U
from spenderq import lyalpha as LyA

from datetime import datetime

import os

import concurrent.futures
import multiprocessing
from tqdm import tqdm
# import traceback

def generate_tmp_deltas(batch_files, batch_start, batch_end):

    #LAMBDA range defined in picca
    picca_lambda_min = 3600.
    picca_lambda_max = 5772.

    #FOREST range (default is LyA)
    forest_lambda_min = 1040.
    forest_lambda_max = 1205.
    
    #wavelength bin is linear, spaced by 0.8
    LAMBDA = np.arange(picca_lambda_min, picca_lambda_max+0.8, 0.8)

    #bad_tids causing issues with picca_cf gathered with help from Andrei
    bad_tids = [64701, 60172026, 210045724, 100326388]
    
    #path to quasar catalog
    quasar_catalogue = '/global/cfs/cdirs/desi/users/abault/spenderq/zcat_hpx_8.fits'

    #open quasar catalog and pull necessary data
    hdul = fits.open(quasar_catalogue)
    catalogue_TID = hdul[1].data['TARGETID']
    catalogue_Z = hdul[1].data['Z']
    catalogue_RA = hdul[1].data['RA']
    catalogue_DEC = hdul[1].data['DEC']
    
    path = '/global/cfs/projectdirs/desi/users/chahah/spender_qso/spenderq_london_v0'
    ## Y1 LONDON MOCKS
    file_prefix = '/DESIlondon_highz.rebin.iter3'

    outpath = '/global/cfs/projectdirs/desi/users/abault/spenderq/deltas_lya/tmp2'
    
    #get all values from spenderq and merge into one array
    normalised_flux, pipeline_weight, z, tid, normalisation, zerr = [], [], [], [], [], []

    sq_reconstructions = []
    
    # print('gathering spenderq batch info')
    for _file in batch_files:

        #load batch
        with open(_file, 'rb') as f:
            _normalised_flux, _pipeline_weight, _z, _tid, _normalisation, _zerr = pickle.load(f)
        _tid = np.array(_tid)
        mask = np.isin(_tid, bad_tids, invert = True)
        if np.count_nonzero(mask) < len(mask):
            print(f'removing bad targetid(s) from {_file}')
        normalised_flux.append(np.array(_normalised_flux)[mask])
        pipeline_weight.append(np.array(_pipeline_weight)[mask])
        z.append(np.array(_z)[mask])
        tid.append(_tid[mask])
        normalisation.append(np.array(_normalisation)[mask])
        zerr.append(np.array(_zerr)[mask])
        
        #load SpenderQ reconstructions
        # _sq_reconstructions = np.load(f'{path}/{file_prefix}_%i.recons.npy' % ibatch)
        filename = os.path.basename(_file)
        name, ext = os.path.splitext(filename)
        _sq_reconstructions = np.load(f'{path}/{name}.recons.npy')
        sq_reconstructions.append(np.array(_sq_reconstructions)[mask])

    normalised_flux=np.concatenate(normalised_flux,axis=0)
    pipeline_weight=np.concatenate(pipeline_weight,axis=0)
    z=np.concatenate(z)
    tid=np.concatenate(tid)
    normalisation=np.concatenate(normalisation,axis=0)
    zerr=np.concatenate(zerr,axis=0)
    sq_reconstructions=np.concatenate(sq_reconstructions,axis=0)
    # print('done gathering spenderq batch info')
    
    nb_of_quasars = len(z)
    # print(f'number of quasars: {nb_of_quasars}')
    
    _tff = np.full((nb_of_quasars,7781),np.nan)
    _weight = np.full((nb_of_quasars,7781),np.nan)
    _sq_cont = np.full((nb_of_quasars,7781),np.nan)
    
    wrecon = np.load(f'{path}/{file_prefix}.wave_recon.npy')
    desi_wave = np.linspace(3600, 9824, 7781) #obs
    
    _tff_bar_up = np.zeros(7781)
    _tff_bar_down = np.zeros(7781)

    #calculate transmitted flux fraction (tff) for these batches
    #for iqso in range(10):
    for iqso in range(nb_of_quasars):
    
        #create wavelength grids
        desi_wave_rest = desi_wave/(1+z[iqso]) #rest
        
        #rebin spender reconstruction
        edges = np.linspace(wrecon[0], wrecon[-1], 7782)
        spenderq_rebin = U.trapz_rebin(wrecon, sq_reconstructions[iqso], edges = edges)
        
        #keep only part of spec that is within the lya range
        mask_desi_rest = (desi_wave_rest >= forest_lambda_min) & (desi_wave_rest <= forest_lambda_max)
    
        _tff[iqso][mask_desi_rest] = normalised_flux[iqso][mask_desi_rest]/spenderq_rebin[mask_desi_rest]
        _weight[iqso][mask_desi_rest] = pipeline_weight[iqso][mask_desi_rest]
        _sq_cont[iqso][mask_desi_rest] = spenderq_rebin[mask_desi_rest]
        _tff_bar_up += normalised_flux[iqso]/spenderq_rebin * pipeline_weight[iqso]
        _tff_bar_down += pipeline_weight[iqso]

    
    
    #get the weighted average transmitted flux fraction 
    picca_range = (desi_wave <= LAMBDA[-1])
    #save _tff parts into numpy array
    filename = f'tff_bar_parts_{batch_end}.npz'
    np.savez(f'{outpath}/{filename}', tff_bar_up = _tff_bar_up, tff_bar_down = _tff_bar_down)
    # print(f'saved tff info to {outpath}/{filename}')
    tff_bar = np.divide(_tff_bar_up[picca_range], _tff_bar_down[picca_range], out=np.zeros_like(_tff_bar_up[picca_range]), where=_tff_bar_down[picca_range]!=0)

    # print('setting up temporary delta files')
    #picca structure: nan filled arrays
    delta = np.full((nb_of_quasars,sum(picca_range)),np.nan)
    _tff_tmp = np.full((nb_of_quasars,sum(picca_range)),np.nan)
    weight = np.full((nb_of_quasars,sum(picca_range)),np.nan)
    sq_cont = np.full((nb_of_quasars,sum(picca_range)),np.nan)

    meta_los_id, meta_ra, meta_dec, meta_z, meta_meansnr, meta_targetid, meta_night, meta_petal, meta_tile = [], [], [], [], [], [], [], [], []

    #for iqso in range(5):
    for iqso in range(nb_of_quasars):
        
        delta[iqso] = (_tff[iqso][picca_range]/tff_bar) - 1. # delta[iqso] = _tff[iqso][picca_range] #but its not actually a delta so call it _tff
        _tff_tmp[iqso] = (_tff[iqso][picca_range])
        weight[iqso] = _weight[iqso][picca_range]
        sq_cont[iqso] = _sq_cont[iqso][picca_range]
        
        los_id_idx = int(np.where(catalogue_TID == int(tid[iqso]))[0][0])
        
        meta_los_id.append(int(catalogue_TID[los_id_idx]))
        meta_ra.append(float(math.radians(catalogue_RA[los_id_idx])))
        meta_dec.append(float(math.radians(catalogue_DEC[los_id_idx])))
        meta_z.append(float(catalogue_Z[los_id_idx]))
        meta_meansnr.append(np.nan)
        meta_targetid.append(int(catalogue_TID[los_id_idx]))
        meta_night.append('')
        meta_petal.append('')
        meta_tile.append('')

    # Create primary HDU 
    primary_hdu = fits.PrimaryHDU(data=None)

    # Create OBSERVED WAVELENGTH HDU
    hdu_wave = fits.ImageHDU(LAMBDA, name=f'LAMBDA')
    hdu_wave.header['HIERARCH WAVE_SOLUTION'] = 'lin'
    hdu_wave.header['HIERARCH DELTA_LAMBDA'] = 0.8

    # Set chunk size, and how to chunk
    chunk_size = 1024

    def chunks(lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    _delta_chunks = chunks(delta, chunk_size)
    # print(np.shape(_delta_chunks))
    _tff_chunks = chunks(_tff_tmp, chunk_size)
    _weight_chunks = chunks(weight, chunk_size)
    _cont_chunks = chunks(sq_cont, chunk_size)

    nb_of_chunks = len(_delta_chunks)
    # print(nb_of_chunks)
    # print('saving temporary delta files')

    i=0
    for ichunk in range(nb_of_chunks):

        # print(i*chunk_size, (i+1)*chunk_size)
        c1 = fits.Column(name='LOS_ID', format='K', array=np.array(meta_los_id[i*chunk_size:(i+1)*chunk_size]))
        c2 = fits.Column(name='RA', format='D', array=np.array(meta_ra[i*chunk_size:(i+1)*chunk_size]))
        c3 = fits.Column(name='DEC', format='D', array=np.array(meta_dec[i*chunk_size:(i+1)*chunk_size]))
        c4 = fits.Column(name='Z', format='D', array=np.array(meta_z[i*chunk_size:(i+1)*chunk_size]))
        c5 = fits.Column(name='MEANSNR', format='D', array=meta_meansnr[i*chunk_size:(i+1)*chunk_size])
        c6 = fits.Column(name='TARGETID', format='K', array=np.array(meta_targetid[i*chunk_size:(i+1)*chunk_size]))
        c7 = fits.Column(name='NIGHT', format='12A', array=meta_night[i*chunk_size:(i+1)*chunk_size])
        c8 = fits.Column(name='PETAL', format='12A', array=meta_petal[i*chunk_size:(i+1)*chunk_size])
        c9 = fits.Column(name='TILE', format='12A', array=meta_tile[i*chunk_size:(i+1)*chunk_size])
        hdu_meta = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9], name='METADATA')
        hdu_meta.header['BLINDING'] = 'none'

        hdu_delta = fits.ImageHDU(_delta_chunks[ichunk], name=f'DELTA')
        hdu_weight = fits.ImageHDU(_weight_chunks[ichunk], name=f'WEIGHT')
        hdu_cont = fits.ImageHDU(_cont_chunks[ichunk], name=f'CONT')
        hdu_tff_bar = fits.ImageHDU(tff_bar, name=f'FBAR')
        hdu_tff_tmp = fits.ImageHDU(_tff_chunks[ichunk], name = f'_TFF_TMP')

        # Combine all HDUs into an HDUList
        hdul = fits.HDUList([primary_hdu, hdu_wave, hdu_meta, hdu_delta, hdu_weight, hdu_cont, hdu_tff_bar, hdu_tff_tmp])

        # Write the HDUList to a new FITS file
        # print(f'{outpath}/delta-%i.fits' % (ichunk+batch_start))
        hdul.writeto(f'{outpath}/delta-%i.fits' % (ichunk+batch_start), overwrite=True)
        i+=1
        
    # print(f'temporary delta files saved to {outpath}.')


def process_batch(batch_files, batch_start, batch_end):
    try:
        generate_tmp_deltas(batch_files, batch_start, batch_end)
        return True
    except Exception as e:
        print(f'Error processing batch: {e}')
        return False




def main():

    # parser = argparse.ArgumentParser( description = 'create temporary delta files from spenderq output', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--batch-start', type = int, required = True, help = 'start value to use for the batch processing')
    # parser.add_argument('--batch-end', type = int, required = True, help = 'end value to use for the batch processing')
    # args = parser.parse_args()

    print(f'start: {datetime.now()}')
    #paths to spender output and delta files output location
    spender_output_path = '/global/cfs/projectdirs/desi/users/chahah/spender_qso/spenderq_london_v0/DESIlondon_highz.rebin.iter3'
    

    ## Y1 LONDON MOCKS
    file_prefix = ''
    # spender_output_files = sorted(glob.glob(path+'/'+file_prefix+'_*.pkl'))
    spender_output_files = sorted(glob.glob(f'{spender_output_path}_*.pkl'))
    
    ## EDR
    #file_prefix = 'DESI.edr.qso_highz'
    #spender_output_files = glob.glob('/global/cfs/projectdirs/desi/users/chahah/spender_qso/DESIedr.qso_highz_*.pkl')

    batch_size = 50
    batches = [spender_output_files[i:i+batch_size] for i in range(0, len(spender_output_files), batch_size)]
    
    # batch_start = args.batch_start
    # batch_end = args.batch_end
    batch_start = [i for i in range(0, len(spender_output_files), batch_size)]
    batch_end = [i+batch_size for i in range(0, len(spender_output_files), batch_size)]

    # batch_files = spender_output_files[batch_start:batch_end]
    
    #determine optimal number of workers based on cpu count
    num_workers = min(multiprocessing.cpu_count(), len(batches))

    print(f"Processing {len(spender_output_files)} files in {len(batches)} batches using {num_workers} workers")

    #track successful and failed batches
    successful_batches = 0
    failed_batches = 0

    #process all batches in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        #submit all batches to the executor
        futures = [executor.submit(process_batch, batch, start, end) for batch, start, end in zip(batches, batch_start, batch_end)]

        #process as they complete with progress tracking
        for future in tqdm(concurrent.futures.as_completed(futures), total = len(batches)):
            try:
                if future.result():
                    successful_batches += 1
                else:
                    failed_batches += 1
            except Exception as e:
                print(f'a batch execution failed: {e}')
                # print(traceback.format_exc())
                failed_batches += 1
    print(f'processing complete: {successful_batches} successful batches, {failed_batches} failed batches')
    print('results were written to disk and can now be processed by your combination code')
    print(f'end: {datetime.now()}')
    # generate_tmp_deltas(batch_files, batch_start, batch_end)

if __name__ == '__main__':
    main()





    