'''

python script for deploying job on adroit 

'''
import os, sys
import time 


def train_spenderq(study, tag, zmin, zmax, debug=False):
    ''' deploy SpenderQ training 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % study,
        ["#SBATCH --time=11:59:59", "#SBATCH --time=00:29:59"][debug], 
        "#SBATCH --export=ALL", 
        "#SBATCH -o o/%s.o" % study, 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --constraint=gpu80", 
        "#SBATCH --mem-per-cpu=8G", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate gqp", 
        "",
        "python /home/chhahn/projects/SpenderQ/bin/train_spender.py /tigress/chhahn/spender_qso/train /tigress/chhahn/spender_qso/models/%s.pt -t %s -n 10 -zmin %f -zmax %f -l 100 -v" % (study, tag, zmin, zmax),
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_spender.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _spender.slurm')
    os.system('rm _spender.slurm')
    return None 


def train_spenderq_simloss(study, tag, zmin, zmax, n_latent=10, similarity=True, consistency=True, debug=False):
    ''' deploy SpenderQ training with similiarty and consistency losses 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % study,
        ["#SBATCH --time=11:59:59", "#SBATCH --time=00:29:59"][debug], 
        "#SBATCH --export=ALL", 
        "#SBATCH -o o/%s.o" % study, 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --constraint=gpu80", 
        "#SBATCH --mem-per-cpu=8G", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate gqp", 
        "", 
        "spender=/home/chhahn/projects/SpenderQ/bin/train_spender_updated.py",
        "python $spender /tigress/chhahn/spender_qso/train /tigress/chhahn/spender_qso/models/%s.pt -t %s -n %i -zmin %f -zmax %f -l 100 -s %s %s" % (study, tag, n_latent, zmin, zmax, ['', '-s'][similarity], ['', '-c'][consistency]),
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_spender.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _spender.slurm')
    os.system('rm _spender.slurm')
    return None 


def postprocess(model_name, input_tag, output_tag, ibatch0, ibatch1, sigma_lya=1.5, sigma_lyb=1.5, wave_lya=1215.67, wave_lyb=1026.00, gpu=True, debug=True): 
    ''' postprocess SpenderQ outputs 
    '''
    # data direcotry
    dir_dat = '/tigress/chhahn/spender_qso/train'

    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s_%s.%i_%i" % (input_tag, output_tag, ibatch0, ibatch1),
        "#SBATCH -o o/%s_%s.%i_%i.o" % (input_tag, output_tag, ibatch0, ibatch1), 
        ["#SBATCH --time=01:59:59", "#SBATCH --time=00:29:59"][debug], 
        "#SBATCH --export=ALL", 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        ['', "#SBATCH --constraint=gpu80"][gpu], 
        "#SBATCH --mem-per-cpu=8G", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate gqp", 
        "",
        "postprocess=/home/chhahn/projects/SpenderQ/bin/postprocess.py", 
        "for ibatch in {%i..%i}; do python $postprocess %s /tigress/chhahn/spender_qso/models/%s.pt -ti %s -to %s -i $ibatch --sigma_lya %f --sigma_lyb %f --wave_lya %f --wave_lyb %f; done" % (ibatch0, ibatch1, dir_dat, model_name, input_tag, output_tag, sigma_lya, sigma_lyb, wave_lya, wave_lyb),
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_spender.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _spender.slurm')
    os.system('rm _spender.slurm')
    return None 


if __name__=="__main__": 
    
    # --- latest run: sim+con losses, SNR scaled rebinning, sigma for LyA and LyB ---
    #postprocess('qso.london.z_2p1_3p5', 'DESIlondon_highz', 'london.lfsc.rsnr.sa1p5sb1p5.i0', 13, 56, sigma_lya=1.5, sigma_lyb=1.5, gpu=True, debug=True)
    #postprocess('qso.london.z_2p1_3p5', 'DESIlondon_highz', 'london.lfsc.rsnr.sa1p5sb1p5.i0', 57, 99, sigma_lya=1.5, sigma_lyb=1.5, gpu=True, debug=True)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p5sb1p5.i0', 'london.lfsc.rsnr.sa1p5sb1p5.i0', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #postprocess('london.lfsc.rsnr.sa1p5sb1p5.i0', 'london.lfsc.rsnr.sa1p5sb1p5.i0', 'london.lfsc.rsnr.sa1p5sb1p5.i1', 0, 49, sigma_lya=1.5, sigma_lyb=1.5, gpu=True, debug=True)
    #postprocess('london.lfsc.rsnr.sa1p5sb1p5.i0', 'london.lfsc.rsnr.sa1p5sb1p5.i0', 'london.lfsc.rsnr.sa1p5sb1p5.i1', 50, 99, sigma_lya=1.5, sigma_lyb=1.5, gpu=True, debug=True)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p5sb1p5.i1', 'london.lfsc.rsnr.sa1p5sb1p5.i1', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #postprocess('london.lfsc.rsnr.sa1p5sb1p5.i1', 'london.lfsc.rsnr.sa1p5sb1p5.i1', 'london.lfsc.rsnr.sa1p5sb1p5.i2', 0, 49, sigma_lya=1.5, sigma_lyb=1.5, gpu=True, debug=True)
    #postprocess('london.lfsc.rsnr.sa1p5sb1p5.i1', 'london.lfsc.rsnr.sa1p5sb1p5.i1', 'london.lfsc.rsnr.sa1p5sb1p5.i2', 50, 99, sigma_lya=1.5, sigma_lyb=1.5, gpu=True, debug=True)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p5sb1p5.i2', 'london.lfsc.rsnr.sa1p5sb1p5.i2', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #postprocess('london.lfsc.rsnr.sa1p5sb1p5.i2', 'london.lfsc.rsnr.sa1p5sb1p5.i2', 'london.lfsc.rsnr.sa1p5sb1p5.i3', 0, 49, sigma_lya=1.5, sigma_lyb=1.5, gpu=True, debug=True)
    #postprocess('london.lfsc.rsnr.sa1p5sb1p5.i2', 'london.lfsc.rsnr.sa1p5sb1p5.i2', 'london.lfsc.rsnr.sa1p5sb1p5.i3', 50, 99, sigma_lya=1.5, sigma_lyb=1.5, gpu=True, debug=True)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p5sb1p5.i3', 'london.lfsc.rsnr.sa1p5sb1p5.i3', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #postprocess('london.lfsc.rsnr.sa1p5sb1p5.i3', 'london.lfsc.rsnr.sa1p5sb1p5.i3', 'london.lfsc.rsnr.sa1p5sb1p5.i4', 0, 49, sigma_lya=1.5, sigma_lyb=1.5, gpu=True, debug=True)
    #postprocess('london.lfsc.rsnr.sa1p5sb1p5.i3', 'london.lfsc.rsnr.sa1p5sb1p5.i3', 'london.lfsc.rsnr.sa1p5sb1p5.i4', 50, 99, sigma_lya=1.5, sigma_lyb=1.5, gpu=True, debug=True)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p5sb1p5.i4', 'london.lfsc.rsnr.sa1p5sb1p5.i4', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    
    # --- fiducial run run: sim+con losses, SNR scaled rebinning, 1.1sigma for LyA and 0.8sigma LyB ---
    #postprocess('qso.london.z_2p1_3p5', 'DESIlondon_highz', 'london.lfsc.rsnr.sa1p1sb0p8.i0', 0, 49, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #postprocess('qso.london.z_2p1_3p5', 'DESIlondon_highz', 'london.lfsc.rsnr.sa1p1sb0p8.i0', 50, 99, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p1sb0p8.i0', 'london.lfsc.rsnr.sa1p1sb0p8.i0', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.i0', 'london.lfsc.rsnr.sa1p1sb0p8.i0', 'london.lfsc.rsnr.sa1p1sb0p8.i1', 0, 49, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.i0', 'london.lfsc.rsnr.sa1p1sb0p8.i0', 'london.lfsc.rsnr.sa1p1sb0p8.i1', 50, 99, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p1sb0p8.i1', 'london.lfsc.rsnr.sa1p1sb0p8.i1', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.i1', 'london.lfsc.rsnr.sa1p1sb0p8.i1', 'london.lfsc.rsnr.sa1p1sb0p8.i2', 0, 49, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.i1', 'london.lfsc.rsnr.sa1p1sb0p8.i1', 'london.lfsc.rsnr.sa1p1sb0p8.i2', 50, 99, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p1sb0p8.i2', 'london.lfsc.rsnr.sa1p1sb0p8.i2', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.i2', 'london.lfsc.rsnr.sa1p1sb0p8.i2', 'london.lfsc.rsnr.sa1p1sb0p8.i3', 0, 49, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.i2', 'london.lfsc.rsnr.sa1p1sb0p8.i2', 'london.lfsc.rsnr.sa1p1sb0p8.i3', 50, 99, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p1sb0p8.i3', 'london.lfsc.rsnr.sa1p1sb0p8.i3', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.i3', 'london.lfsc.rsnr.sa1p1sb0p8.i3', 'london.lfsc.rsnr.sa1p1sb0p8.i4', 0, 10, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.i3', 'london.lfsc.rsnr.sa1p1sb0p8.i3', 'london.lfsc.rsnr.sa1p1sb0p8.i4', 50, 74, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.i3', 'london.lfsc.rsnr.sa1p1sb0p8.i3', 'london.lfsc.rsnr.sa1p1sb0p8.i4', 75, 99, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #for iseed in range(1, 5): 
    #    train_spenderq_simloss('london.lfsc.rsnr.sa1p1sb0p8.i3.%i' % iseed, 'london.lfsc.rsnr.sa1p1sb0p8.i3', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p1sb0p8.no_bal_dla.i3', 'london.lfsc.rsnr.sa1p1sb0p8.no_bal_dla.i3', 2.1, 3.5, similarity=True, consistency=True, debug=False)

    # --- sigma tests --- 
    #postprocess('london.rebin.sigma2.iter3', 'london.rebin.sigma2.iter3', 'london.rebin.sigma2.iter4', 0, 5, sigma=2., gpu=True, debug=True)
    #
    #postprocess('london.rebin.sigma1.iter3', 'london.rebin.sigma1.iter3', 'london.rebin.sigma1.iter4', 0, 5, sigma=1., gpu=True, debug=True)
    
    #train_spenderq('london.rebin.sigma2.iter3', 'london.rebin.sigma2.iter3', 2.1, 3.5, debug=False)
    #train_spenderq('london.rebin.sigma1.iter3', 'london.rebin.sigma1.iter3', 2.1, 3.5, debug=False)
    
    # --- similiarity and consistency loss test --- 
    #train_spenderq_simloss('london.rebin.iter3.simloss', 'DESIlondon_highz.rebin.iter3',
    #            2.1, 3.5, n_latent=10, similarity=True, consistency=True, debug=False)
    #for iseed in range(5): 
    #    train_spenderq_simloss('london.rebin.iter3.simconloss.%i' % iseed, 'DESIlondon_highz.rebin.iter3',
    #                2.1, 3.5, n_latent=10, similarity=True, consistency=True, debug=False)
    
    # --- nlatent test --- 
    #for nlatent in [6, 8, 12]: 
    #    train_spenderq_simloss('london.rebin.iter3.simconloss.nlatent%i' % nlatent, 'DESIlondon_highz.rebin.iter3',
    #                2.1, 3.5, n_latent=nlatent, similarity=True, consistency=True, debug=False)

    # --- high LyA upper bound run: sim+con losses, SNR scaled rebinning, 1.1sigma for LyA and 0.8sigma LyB ---
    #postprocess('qso.london.z_2p1_3p5', 'DESIlondon_highz', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i0', 0, 49, sigma_lya=1.1, sigma_lyb=0.8, wave_lya=1250., gpu=True, debug=True)
    #postprocess('qso.london.z_2p1_3p5', 'DESIlondon_highz', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i0', 50, 99, sigma_lya=1.1, sigma_lyb=0.8, wave_lya=1250., gpu=True, debug=True)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p1sb0p8.hilya.i0', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i0', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.hilya.i0', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i0', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i1', 0, 49, sigma_lya=1.1, sigma_lyb=0.8, wave_lya=1250., gpu=True, debug=True)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.hilya.i0', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i0', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i1', 50, 99, sigma_lya=1.1, sigma_lyb=0.8, wave_lya=1250., gpu=True, debug=True)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p1sb0p8.hilya.i1', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i1', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.hilya.i1', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i1', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i2', 0, 49, sigma_lya=1.1, sigma_lyb=0.8, wave_lya=1250., gpu=True, debug=True)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.hilya.i1', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i1', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i2', 50, 99, sigma_lya=1.1, sigma_lyb=0.8, wave_lya=1250., gpu=True, debug=True)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p1sb0p8.hilya.i2', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i2', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.hilya.i2', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i2', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i3', 0, 49, sigma_lya=1.1, sigma_lyb=0.8, wave_lya=1250., gpu=True, debug=True)
    #postprocess('london.lfsc.rsnr.sa1p1sb0p8.hilya.i2', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.ik', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i3', 50, 99, sigma_lya=1.1, sigma_lyb=0.8, wave_lya=1250., gpu=True, debug=True)
    train_spenderq_simloss('london.lfsc.rsnr.sa1p1sb0p8.hilya.i3', 'london.lfsc.rsnr.sa1p1sb0p8.hilya.i3', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #for iseed in range(1, 5): 
    #    train_spenderq_simloss('london.lfsc.rsnr.sa1p1sb0p8.i3.%i' % iseed, 'london.lfsc.rsnr.sa1p1sb0p8.i3', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p1sb0p8.no_bal_dla.i3', 'london.lfsc.rsnr.sa1p1sb0p8.no_bal_dla.i3', 2.1, 3.5, similarity=True, consistency=True, debug=False)


