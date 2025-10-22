'''

python script for deploying job on NERSC Perlmutter 

'''
import os, sys
import time 


def train_spenderq(study, tag, zmin=2.1, zmax=3.5, n_latent=10, similarity=True, consistency=True, debug=False):
    ''' deploy SpenderQ training with similiarty and consistency losses 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -A desi", 
        "#SBATCH -C gpu", 
        "#SBATCH -q shared", 
        "#SBATCH -J %s" % study,
        "#SBATCH -o o/%s.o" % study, 
        ["#SBATCH --time=11:59:59", "#SBATCH --time=00:29:59"][debug], 
        "#SBATCH -n 1", 
        "#SBATCH -c 32", 
        "#SBATCH --gpus-per-task=1", 
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK", 
        "export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK", 
        'export SLURM_CPU_BIND="cores"', 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate spenderq", 
        "", 
        "spender=/global/homes/c/chahah/projects/SpenderQ/bin/train_spender_updated.py",
        "dir_train=/global/cfs/projectdirs/desi/users/chahah/spender_qso/train/dr1", 
        "outfile=/global/cfs/projectdirs/desi/users/chahah/spender_qso/models/dr1/%s.pt" % study, 
        "srun python $spender $dir_train $outfile -t %s -n %i -zmin %f -zmax %f -l 100 -s %s %s" % (tag, n_latent, zmin, zmax, ['', '-s'][similarity], ['', '-c'][consistency]),
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_spender.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _spender.slurm')
    #os.system('rm _spender.slurm')
    return None 


def postprocess(model_name, input_tag, output_tag, ibatch0, ibatch1, sigma_lya=1.5, sigma_lyb=1.5, gpu=True, debug=True): 
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
        "for ibatch in {%i..%i}; do python $postprocess %s /tigress/chhahn/spender_qso/models/%s.pt -ti %s -to %s -i $ibatch --sigma_lya %f --sigma_lyb %f; done" % (ibatch0, ibatch1, dir_dat, model_name, input_tag, output_tag, sigma_lya, sigma_lyb),
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


def _postprocess(model_name, input_tag, output_tag, ibatch0, ibatch1, sigma=1.5, debug=True): 
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
        "postprocess=/home/chhahn/projects/SpenderQ/bin/postprocess.py", 
        "for ibatch in {%i..%i}; do python $postprocess %s /tigress/chhahn/spender_qso/models/%s.pt -ti %s -to %s -i $ibatch -sigma %f; done" % (ibatch0, ibatch1, dir_dat, model_name, input_tag, output_tag, sigma),
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


def absorbers(absorber, model_tag, input_tag, ibatch, n_model=5, debug=True): 
    ''' find aborbers using the reconstructions 
    '''
    # data direcotry
    dir_dat = '/tigress/chhahn/spender_qso/absorbers'

    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s_%s.%i" % (absorber, input_tag, ibatch),
        "#SBATCH -o o/%s_%s.%i.o" % (absorber, input_tag, ibatch),
        ["#SBATCH --time=01:59:59", "#SBATCH --time=00:29:59"][debug], 
        "#SBATCH --export=ALL", 
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
        "find_absorbers=/home/chhahn/projects/SpenderQ/bin/find_absorbers.py", 
        "python $find_absorbers %s %s -m %s -nm %i -ti %s -i %i" % (dir_dat, absorber, model_tag, n_model, input_tag, ibatch),
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
    # edr 
    train_spenderq('qso.dr1.hiz', 'DESI.dr1.qsohiz', similarity=True, consistency=True, debug=True) # first spenderq training with sim and con loss
    
    #postprocess('qso.edr.hiz', 'DESI.edr.qso_highz', 'edr.lfsc.rsnr.sa1p1sb0p8.i0', 0, 42, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #postprocess('qso.edr.hiz', 'DESI.edr.qso_highz', 'edr.lfsc.rsnr.sa1p1sb0p8.i0', 43, 84, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #train_spenderq('qso.edr.hiz.sa1p5sb1p5.i0', 'edr.lfsc.rsnr.sa1p1sb0p8.i0', 2.1, 3.5, similarity=True, consistency=True, debug=False) # first spenderq training with sim and con loss
    #postprocess('qso.edr.hiz.sa1p5sb1p5.i0', 'edr.lfsc.rsnr.sa1p1sb0p8.i0', 'edr.lfsc.rsnr.sa1p1sb0p8.i1', 0, 42, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #postprocess('qso.edr.hiz.sa1p5sb1p5.i0', 'edr.lfsc.rsnr.sa1p1sb0p8.i0', 'edr.lfsc.rsnr.sa1p1sb0p8.i1', 43, 84, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #train_spenderq('qso.edr.hiz.sa1p5sb1p5.i1', 'edr.lfsc.rsnr.sa1p1sb0p8.i1', 2.1, 3.5, similarity=True, consistency=True, debug=False) # first spenderq training with sim and con loss
    #postprocess('qso.edr.hiz.sa1p5sb1p5.i1', 'edr.lfsc.rsnr.sa1p1sb0p8.i1', 'edr.lfsc.rsnr.sa1p1sb0p8.i2', 0, 42, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #postprocess('qso.edr.hiz.sa1p5sb1p5.i1', 'edr.lfsc.rsnr.sa1p1sb0p8.i1', 'edr.lfsc.rsnr.sa1p1sb0p8.i2', 43, 84, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #train_spenderq('qso.edr.hiz.sa1p5sb1p5.i2', 'edr.lfsc.rsnr.sa1p1sb0p8.i2', 2.1, 3.5, similarity=True, consistency=True, debug=False) # first spenderq training with sim and con loss
    #postprocess('qso.edr.hiz.sa1p5sb1p5.i2', 'edr.lfsc.rsnr.sa1p1sb0p8.i2', 'edr.lfsc.rsnr.sa1p1sb0p8.i3', 0, 42, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #postprocess('qso.edr.hiz.sa1p5sb1p5.i2', 'edr.lfsc.rsnr.sa1p1sb0p8.i2', 'edr.lfsc.rsnr.sa1p1sb0p8.i3', 43, 84, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #train_spenderq('qso.edr.hiz.sa1p5sb1p5.i3', 'edr.lfsc.rsnr.sa1p1sb0p8.i3', 2.1, 3.5, similarity=True, consistency=True, debug=False) # first spenderq training with sim and con loss
    #for iseed in range(1,5): 
    #    train_spenderq('qso.edr.hiz.sa1p5sb1p5.i3.%i' % iseed, 'edr.lfsc.rsnr.sa1p1sb0p8.i3', 2.1, 3.5, similarity=True, consistency=True, debug=False) # first spenderq training with sim and con loss
    #postprocess('qso.edr.hiz.sa1p5sb1p5.i3', 'edr.lfsc.rsnr.sa1p1sb0p8.i3', 'edr.lfsc.rsnr.sa1p1sb0p8.i4', 0, 42, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #postprocess('qso.edr.hiz.sa1p5sb1p5.i3', 'edr.lfsc.rsnr.sa1p1sb0p8.i3', 'edr.lfsc.rsnr.sa1p1sb0p8.i4', 43, 84, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    
    #train_spenderq_simloss('london.lfsc.rsnr.sa1p5sb1p5.i0', 'london.lfsc.rsnr.sa1p5sb1p5.i0', 2.1, 3.5, similarity=True, consistency=True, debug=False)
    #postprocess('qso.edr.z_2p1_3p5', # model name
    #        'DESIedr.qso_highz', 
    #        'edr.highz.iter0', 
    #        0, 27, sigma=1.5, debug=True) 
    #train_spenderq('edr.highz.iter0', 'edr.highz.iter0', 2.1, 3.5, debug=False)
    #postprocess('edr.highz.iter0', 'edr.highz.iter0',  'edr.highz.iter1', 0, 27, sigma=1.5, debug=True) 
    #train_spenderq('edr.highz.iter1', 'edr.highz.iter1', 2.1, 3.5, debug=False)
    #postprocess('edr.highz.iter1', 'edr.highz.iter1',  'edr.highz.iter2', 0, 27, sigma=1.5, debug=True) 
    #train_spenderq('edr.highz.iter2', 'edr.highz.iter2', 2.1, 3.5, debug=False)
