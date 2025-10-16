import numpy as np
import glob
from astropy.io import fits
import os

from datetime import datetime

def main():
    
    print(f'start: {datetime.now()}')
    
    path = '/global/cfs/cdirs/desi/users/abault/spenderq/deltas_lya/tmp2'
    outpath = '/global/cfs/cdirs/desi/users/abault/spenderq/deltas_lya/Delta/test3'
    
    #set desi and picca related variables 
    desi_wave = np.linspace(3600, 9824, 7781)
    picca_lambda_min = 3600.
    picca_lambda_max = 5772.
    LAMBDA = np.arange(picca_lambda_min, picca_lambda_max+0.8, 0.8)
    picca_range = (desi_wave <= LAMBDA[-1])
    MIN_FOREST_LENGTH = 150

    #find all files for _tff and deltas
    _tff_files = glob.glob(f'{path}/*.npz')
    delta_files = glob.glob(f'{path}/*.fits')

    _tff_bar_up = 0
    _tff_bar_down = 0

    #get the tff_bar value for the entire
    for file in _tff_files:
        _file = np.load(file)
        _tff_bar_up += _file['tff_bar_up']
        _tff_bar_down += _file['tff_bar_down']

    tff_bar = np.divide(_tff_bar_up[picca_range], _tff_bar_down[picca_range], out=np.zeros_like(_tff_bar_up[picca_range]), where=_tff_bar_down[picca_range]!=0)

    print(f'updating {len(delta_files)} files.')
    count = 0
    for dfile in delta_files:
        filename = os.path.basename(dfile)
        f = fits.open(dfile)
        new_delta = (f['_TFF_TMP'].data/tff_bar) - 1.
        #create mask to remove forests that are too short
        mask = np.ones(len(new_delta), dtype = bool)
        for i,row in enumerate(new_delta):
            len_delta = np.count_nonzero(~np.isnan(row))
            if len_delta < MIN_FOREST_LENGTH:
                mask[i] = False
                print(f'removed forest from {dfile} due to too short: {len_delta}')
        #need to remove masks from appropriate places (deltas, weight, cont, metadata)
        f['DELTA'].data = new_delta[mask]
        f['WEIGHT'].data = f['WEIGHT'].data[mask]
        f['CONT'].data = f['CONT'].data[mask]
        f['METADATA'].data = f['METADATA'].data[mask]
        f['FBAR'].data = tff_bar
        del f['_TFF_TMP']
        f.writeto(f'{outpath}/{filename}', overwrite = True)
        f.close()
        
        count += 1
        if count % 50 == 0:
            print(np.round(count/len(delta_files)*100,2))
    print(f'wrote files to {outpath}')
    print(f'end: {datetime.now()}')
    
if __name__ == '__main__':
    main()